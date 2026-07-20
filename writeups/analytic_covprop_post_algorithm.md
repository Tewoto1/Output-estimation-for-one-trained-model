# Analytic covprop — pure algorithm sheet (`analytic_kprop`, `fit="post"`)

Predicts `E[f(X)]`, `X ~ N(0, s²Iₙ)`, for a ReLU MLP whose hidden matrices carry a
coordinate spike, `M = W + e₁e₁ᵀ`. This sheet is the **post-activation-fit variant
only** (`fit="post"` — the accurate one), exactly as implemented in
`Mecha_preds/analytic_kprop/core.py::analytic_layer_update_post` /
`run_analytic_kprop_k2(fit="post")`. Formulas are the implemented ones; per-step
runtime in each heading. Companion docs: `Mecha_preds/analytic_kprop/ALGORITHM.md`
(both variants, prose), `writeups/analytic_kprop_runtime_analysis.md` (measurements).

## Notation

```
n = width, d = n − 1.  X = (A, B): spike A = X₀ (direction e₁), bulk B ∈ ℝᵈ.
Layer block of (M, b):  γ = M₀₀, r = M₀,₁:, u = M₁:,₀, V = M₁:,₁:, β = b₀, η = b₁:
Pre-activations:        Y = γA + rᵀB + β   (new spike),   C = uA + VB + η   (new bulk)
K = incoming components, J = retained cells this layer, T = worker slots.
Φ, φ = normal cdf/pdf;  Φ₂(·,·;ρ), φ₂ = bivariate cdf (Owen's T) / pdf;  sym(A) = (A+Aᵀ)/2.
c_sf = special-function cost per matrix entry (≈100–300 ns: two owens_t + several Φ/φ).
```

## State (`PostAffineState`) — memory O(d²)

```
nodes  {p_k, a_k}, k = 0..K−1          (node 0 = zero atom, a₀ = 0, when present)
positive branch (a > 0):   E[B|A=a]  ≈ m(a) = m0 + m1·a
                           Cov[B|A=a] ≈ S(a) = W0 + W1·a
zero atom:  exact (m_at, S_at)         [atom="exact", default; None ⇒ atom uses the family]
t2_k = Var(A | k):  0 except the input component
```

Input state (exact — no input discretization): `K=1, p=1, a=0, t2=s², m0=m1=0,
W0=s²I_d, W1=0`, no atom. Layer 1 therefore starts from the exact input law.

---

## One hidden layer (`analytic_layer_update_post`)

### 1. Family transform = the whole linear step — **2–3 congruences O(d³) + 4–6 mat-vecs O(d²)**, independent of J

```
c0 = V m0 + η          c1 = u + V m1
g0 = V (W0 r)          g1 = V (W1 r)
s0 = rᵀ W0 r           s1 = rᵀ W1 r
G0 = V W0 Vᵀ           G1 = V W1 Vᵀ
atom:  mC_at = V m_at + η,   g_at = V (S_at r),   s_at = rᵀ S_at r,   G_at = V S_at Vᵀ
```

Linear maps preserve affinity exactly, so this transforms the four family objects
instead of per-node moments. Every component's (Y, C) parameters are then closed-form
affine in `a_k` (O(K) scalars):

```
non-atom k:   m_Y,k  = (β + rᵀm0) + (γ + rᵀm1)·a_k
              s²_Y,k = s0 + s1·a_k + γ²·t2_k                     (clipped ≥ 0)
              m_C,k  = c0 + c1·a_k
              g_k    = Cov(C,Y|k) = g0 + a_k·g1 + γ·t2_k·u
              S_C,k  = Cov(C|k)   = G0 + a_k·G1 + t2_k·uuᵀ       (never materialized)
atom:         m_Y = β + rᵀm_at,   s²_Y = s_at,   m_C = mC_at,   g = g_at,   S_C = G_at
```

The congruences are the only O(d³) work; `device="cuda"` offloads exactly these.

### 2. Scalar grid — **O(I·K·J_cells) scalars, I ≤ 1000 Lloyd iterations** (≈0.1 s, d-independent)

`Y` is the known mixture `ν_Y = Σ_k p_k N(m_Y,k, s²_Y,k)`. Sign-split budget:

```
neg   = Σ_k p_k Φ(−m_Y,k / s_Y,k)                     (mass of {Y ≤ 0}, closed form)
n_pos = num_nodes                                     (the hyperparameter)
n_neg = clip( ⌈n_pos · neg/(1−neg)⌉ , 1, 8·n_pos )
```

Edges (`grid="w2"`): Lloyd–Max (W2-optimal) quantization of the exact mixture, run
per sign — one edge **exactly at 0**, outer edges ±∞, `J_cells = n_neg + n_pos`.
Only positive cells survive the ReLU, so the positive side always gets the full
budget; the negative side matches its mass-per-cell and later collapses into the atom.

### 3. Pair stats — **O(K·J_cells) special functions** (one (K, J) broadcast)

For stochastic k (`s²_Y,k ≥ min_prob`) and cell `j = [e_j, e_{j+1})`, with
`α = (e_j − m_Y,k)/s_Y,k`, `β̃ = (e_{j+1} − m_Y,k)/s_Y,k` (truncated-normal identities):

```
Q_kj = Φ(β̃) − Φ(α)                                          P(Y ∈ cell j | k)
δ_kj = s_Y,k (φ(α) − φ(β̃)) / Q_kj                           E[Y | k, j] − m_Y,k
v_kj = s²_Y,k [ 1 + (α φ(α) − β̃ φ(β̃))/Q_kj ] − δ²_kj  (≥0)  Var(Y | k, j)
```

Entries with `Q ≤ min_prob` are zeroed. A deterministic component (`s²_Y < min_prob`)
puts its whole mass in its containing cell with `δ = v = 0`.

### 4. Cell masses / centroids / posteriors — **O(K·J)**

```
w̃_j = Σ_k p_k Q_kj      W_tot = Σ_j w̃_j       drop cells with w̃_j ≤ min_prob  → J kept
η_k|j = p_k Q_kj / w̃_j        y_j = Σ_k η_k|j (m_Y,k + δ_kj)        w_j = w̃_j / W_tot
```

Masses and centroids are exact for any grid; `max(0, 1 − W_tot)` is logged as `mass_lost`.

### 5. Exact cell reconditioning in a 7-vector basis — **O(K·J·7²) scalars + O(J·d)**; nothing d×d per (k, j)

Within component k, (Y, C) is jointly Gaussian ⇒ conditioning on `Y ∈ cell j` is
exact Gaussian regression + the step-3 truncated moments:

```
m_k→j = m_C,k + (δ_kj / s²_Y,k) · g_k
S_k→j = S_C,k + h_kj · g_k g_kᵀ ,          h_kj = (v_kj − s²_Y,k) / s⁴_Y,k
```

All vectors lie in `span(B)`, `B = [c0 | c1 | mC_at | g0 | g1 | g_at | u] ∈ ℝ^{d×7}`,
with closed-form scalar coordinates (`ℓ = δ_kj / s²_Y,k`; columns indexed 1–7):

```
m_k→j = B·w_kj :   non-atom  w_kj = (1, a_k, 0, ℓ, a_k·ℓ, 0, γ·t2_k·ℓ)
                   atom      w_kj = (0, 0, 1, 0, 0, ℓ, 0)
g_k   = B·gc_k :   non-atom  gc_k = (0, 0, 0, 1, a_k, 0, γ·t2_k)
                   atom      gc_k = (0, 0, 0, 0, 0, 1, 0)
```

Cell merge (total expectation / total covariance ⇒ `Ŝ_j` **PSD by construction**):

```
w̄_j = Σ_k η_k|j w_kj                     m̂_j = B w̄_j
C_j = Σ_k η_k|j (w_kj − w̄_j)(w_kj − w̄_j)ᵀ  +  Σ_k η_k|j h_kj gc_k gc_kᵀ  +  (Σ_k η_k|j t2_k) e₇e₇ᵀ
α_j = Σ_{k≠atom} η_k|j        β_j = Σ_{k≠atom} η_k|j a_k        ω_j = η_atom|j
Ŝ_j = α_j G0 + β_j G1 + ω_j G_at + B C_j Bᵀ          (assembled per node in step 6)
```

The reconditioned pre-activation is **never projected** — cells feed the ReLU exactly;
the only fit in this variant happens after the ReLU (step 8).

### 6. Exact Gaussian→ReLU per cell, streaming — **J·(c_sf·d² kernel + O(d²) assembly)** = **≥93% of total runtime**; memory O(T·d²)

Thread slot t owns cells `j ≡ t (mod T)`. Per cell: assemble `Ŝ_j` dense from the
step-5 factored form, floor its diagonal at 0 (roundoff guard only — no
factorization needed, `Ŝ_j` is a mixture covariance), then exact rectified-Gaussian
moments of `Z⁺ = ReLU(Z)`, `Z ~ N(m̂_j, Ŝ_j)` (`_utils.exact_relu_covariance`), with
`α_i = μ_i/σ_i`:

```
mean:      r_i  = μ_i Φ(α_i) + σ_i φ(α_i)
diagonal:  R_ii = (μ_i² + σ_i²) Φ(α_i) + μ_i σ_i φ(α_i) − r_i²
off-diag (ρ = Ŝ_ab/σ_aσ_b):   A = φ(α_a) Φ((α_b − ρα_a)/√(1−ρ²)),   B = φ(α_b) Φ((α_a − ρα_b)/√(1−ρ²))
  E[Z_a⁺ Z_b⁺] = μ_aμ_b Φ₂(α_a,α_b;ρ) + μ_aσ_b (B + ρA) + μ_bσ_a (A + ρB)
               + σ_aσ_b [ ρΦ₂ + (1−ρ²) φ₂ − ρα_a A − ρα_b B ]
  R_ab = E[Z_a⁺ Z_b⁺] − r_a r_b
```

(Degenerate coords and `|ρ| ≈ 1` entries use dedicated closed forms inside the kernel.)
`r_j` is stored (J×d); `R_j` is **never stored** — each slot streams it into its own
accumulators, summed in fixed slot order at the end:

```
AR⁺ = Σ_{y_j>0} w_j R_j      AyR⁺ = Σ_{y_j>0} w_j y_j R_j
AR⁻ = Σ_{y_j≤0} w_j R_j      F2   = Σ_{y_j>0} w_j ‖R_j‖²_F
```

Deterministic per worker count; fp-identical to serial only at T = 1 (else allclose).

### 7. Zero atom: exact merge of all `y_j ≤ 0` cells — **O(J⁻·d²)**

```
p0 = Σ_{y_j≤0} w_j ,    z_j = w_j / p0
m_at' = Σ z_j r_j
S_at' = AR⁻/p0 + Σ z_j (r_j − m_at')(r_j − m_at')ᵀ
```

Total expectation / total covariance of the **post-ReLU** moments (never a pre-ReLU
Gaussian merge).

### 8. Post-ReLU affine fit — the variant's single projection — **O(J⁺·d²) (residual outer products) + O(d²)**

Data: positive cells `{(w_j, y_j, r_j)}` + the streamed `AR⁺, AyR⁺`. `atom="fit"`
appends `(p0, a=0, m_at')` and `AR⁺ += p0·S_at'`; `atom="exact"` (default) keeps the
atom out. Normalize weights to `fw` (total `m_f`), `AR = AR⁺/m_f`, `AyR = AyR⁺/m_f`.
Weighted least squares of `m(a)` and `S(a)` on the node data (abscissae
`a_j = y_j` for cells, `0` for the appended atom point):

```
ā = Σ fw_j a_j          v_a = Σ fw_j (a_j − ā)²          (v_a ≤ 1e-14 ⇒ slopes = 0)
m1' = Σ fw_j (a_j − ā) r_j / v_a          m0' = Σ fw_j r_j − ā·m1'
W1' = sym( (AyR − ā·AR) / v_a )           W0' = sym( AR − ā·W1' )
e_j = r_j − m0' − a_j m1'      E_m = Σ fw_j ‖e_j‖²      R_m = Σ fw_j e_j e_jᵀ
cov_intercept="mc" (default):   W0' += R_m
```

LS orthogonality conserves the weighted mean exactly ⇒ the depth-1 readout mean is
pure quadrature error (≈4e-8 at budget 40). The `mc` intercept adds the between-node
mean-residual covariance back so the unconditional bulk second moment is conserved.
Streaming covariance residual (diagnostic, vs the LS intercept:
`W0_ls = W0' − R_m` under `"mc"`, `= W0'` under `"ls"`; all Frobenius products, O(d²)):

```
E_S = Σ fw_j ‖R_j − W0_ls − a_j W1'‖²_F
    = F2/m_f − 2⟨AR, W0_ls⟩ − 2⟨AyR, W1'⟩ + ‖W0_ls‖² + 2ā⟨W0_ls, W1'⟩ + (Σ fw_j a_j²)‖W1'‖²
```

(`atom="fit"` adds `(p0/m_f)·‖S_at'‖²` to `F2/m_f`.) If every cell died (`m_f = 0`):
family = 0. Full derivation of this step: appendix A.

### 9. Next state — **O(J)**

```
p = normalize([p0 ; w_{y>0}])       a = [0 ; y_{y>0}]       family (m0', m1', W0', W1')
atom="exact" and p0 > 0  ⇒  keep (m_at', S_at') exact;  else atom_m = atom_S = None
```

Per-layer logs: `mass_lost`, `E_m`, `E_S`, `tr R_m`, `psd_clipped` (diagonal floor
mass), `num_cells`, `num_pos_nodes`, `zero_atom_mass`, and the quantization
distortion `Σ_kj p_k Q_kj [(m_Y,k + δ_kj − y_j)² + v_kj] / W_tot`, plus phase timers
`stats["t_*"]`.

---

## Readout (after the last hidden layer) — O(n·out)

```
E[f(X)] = W_ro · ( Σ_k p_k a_k  ;  Σ_k p_k m_k ) + b_ro ,
          m_k = m_at (exact atom)  else  m0 + m1·a_k
```

## Approximation ledger

Exact: input layer (t2 carries Var(A), no discretization); family transform (step 1);
cell masses/centroids (4); within-component reconditioning (5); per-cell ReLU
moments (6); atom merge (7); readout mean given the state.
Approximate: **(i)** conditional K = 2 closure — each cell is summarized by
`(m̂_j, Ŝ_j)` and re-Gaussianized at the ReLU input; **(ii)** the post-ReLU affine
projection (8) — `r(a), R(a)` are nonlinear in `a` (Φ factors, strongest near
`a ≈ 0`, hence the exact-atom default); **(iii)** the 1-D quantization term,
controlled by `num_nodes` (knee ≈ 6–10) and logged as `scalar_distortion`.

## Knobs (defaults)

`num_nodes=40` (positive-side cell budget), `grid="w2"`, `bulk_relu="exact"`,
`cov_intercept="mc"`, `atom="exact"`, `min_prob=1e-15`, `workers=auto`
(env `BINNED_KPROP_WORKERS`), `device=None` (`"cuda"` offloads the step-1
congruences only; the ReLU kernel stays on CPU — threading is what accelerates it).

## Per-layer complexity summary

| step | cost | dominant object |
|---|---|---|
| 1 family transform | O(d³) + O(d²), fixed | 2–3 congruences `V·Vᵀ` |
| 2 grid | O(I·K·J) scalars | Lloyd–Max, ≈0.1 s, d-independent |
| 3 pair stats | O(K·J) sf | truncated-normal Φ/φ |
| 4 masses/centroids | O(K·J) | — |
| 5 recondition (7-basis) | O(K·J·49) + O(J·d) | (J,7,7) coefficient tensors |
| 6 ReLU kernel | **J·c_sf·d²** + O(J·d²) assembly | owens_t ≈ 60% of total program time |
| 7 atom merge | O(J⁻·d²) | — |
| 8 post-fit | O(J⁺·d²) + O(d²) | `R_m` outer products |
| **total** | **Θ(J·c_sf·d² + d³)** | kernel ≥ 93% at every width |
| state memory | **O(d²)** (+ O(T·d²) transient) | no (J, d, d) anywhere |

Measured (depth 2, budget 40, 4-core sandbox): totals 0.21 / 0.31 / 1.76 / 4.85 /
18.2 s at n = 128 / 256 / 512 / 1024 / 2048; runs at n = 2048 in ≈2 GB where
`fit="pre"` is OOM-killed (its (J, d, d) stack). Linear in `num_nodes` past fixed
overheads. To go faster: attack the kernel (upper-triangle-only ≈1.7×), lower
`num_nodes` toward the knee, or `bulk_relu="gain"` — not the propagation.
Why no per-bin d³ term exists at all: appendix B.

---

## Appendix A — deriving the affine fit (step 8)

**Data.** After the ReLU, the fit sees nodes `j` with normalized weights `fw_j`
(`Σ fw_j = 1`), abscissae `a_j` (= `y_j` for positive cells; `0` for the atom point
when `atom="fit"`), responses `r_j ∈ ℝᵈ` (post-ReLU means) and `R_j ∈ ℝ^{d×d}`
(post-ReLU covariances). The model is the affine family

```
m(a) = m0 + a·m1 ,        S(a) = W0 + a·W1 .
```

**Objective.** Weighted least squares, Euclidean for the mean and Frobenius for the
covariance:

```
L_m(m0, m1) = Σ_j fw_j ‖ r_j − m0 − a_j m1 ‖²
L_S(W0, W1) = Σ_j fw_j ‖ R_j − W0 − a_j W1 ‖²_F
```

**Decoupling.** Both norms are sums of squares over entries, and the model touches
each entry independently: entry `c` of `(m0, m1)` only meets entry `c` of the data,
entry `(a,b)` of `(W0, W1)` only meets `R_j,ab`. So `L_m` splits into `d` — and `L_S`
into `d²` — **independent copies of the same scalar problem**, all sharing one design
`{(fw_j, a_j)}`:

```
min_{β0, β1}  Σ_j fw_j ( t_j − β0 − β1 a_j )² .
```

**Normal equations and solution.** Setting the two gradients to zero:

```
Σ_j fw_j ( t_j − β0 − β1 a_j )      = 0        (intercept equation)
Σ_j fw_j a_j ( t_j − β0 − β1 a_j )  = 0        (slope equation)
```

With `ā = Σ fw a`, `t̄ = Σ fw t`, `v_a = Σ fw (a − ā)²`, subtracting `ā ×` the first
from the second gives the closed form (the shared 2×2 normal matrix
`[[1, ā], [ā, Σfw a²]]` is inverted once, by centering):

```
β1 = Σ_j fw_j (a_j − ā) t_j / v_a ,        β0 = t̄ − β1 ā .
```

Substituting `t_j → r_j` entrywise gives `m1', m0'` of step 8. Substituting
`t_j → R_j,ab` and collecting all `d²` entries into matrices:

```
W1' = Σ_j fw_j (a_j − ā) R_j / v_a  =  (AyR − ā·AR) / v_a ,      W0' = AR − ā·W1' ,
```

since `Σ fw (a−ā) R = Σ fw a R − ā Σ fw R = AyR − ā·AR`. **Sufficient statistics:**
the solution depends on the `R_j` only through `AR = Σ fw R_j` and
`AyR = Σ fw a_j R_j` — which is exactly why they can be streamed inside the ReLU loop
and the `(J, d, d)` stack never exists. `v_a ≤ 1e-14` ⇒ slope unidentifiable ⇒
`β1 = 0, β0 = t̄` (single surviving node). The `sym(·)` in the code is cosmetic:
symmetric data ⇒ symmetric solution; it only kills roundoff.

**Conservation identities.** The intercept equation, summed over entries, says the
fit reproduces the weighted data means *exactly*:

```
Σ_j fw_j m(a_j) = Σ_j fw_j r_j          Σ_j fw_j S_ls(a_j) = Σ_j fw_j R_j .
```

The first is why the depth-1 readout mean is pure quadrature error: the bulk readout
uses `Σ_k p_k m_k`, which by this identity equals `Σ_j w_j r_j` over the very cells
the fit was built on (the exact atom contributes `m_at'` verbatim).

**The `mc` intercept `W0' += R_m`.** Compare unconditional bulk second moments over
the node law (law of total covariance), exact mixture vs fitted family:

```
Cov_exact(B) = Σ fw R_j        + Σ fw (r_j − r̄)(r_j − r̄)ᵀ                 (r̄ = Σ fw r_j)
Cov_fit(B)   = Σ fw S_ls(a_j)  + Σ fw (m(a_j) − r̄)(m(a_j) − r̄)ᵀ
```

Within parts are equal (second identity above). For the between parts write
`r_j = m(a_j) + e_j` and note `m(a_j) − r̄ = (a_j − ā)·m1'`; the residual
orthogonality from the two normal equations, `Σ fw e_j = 0` and `Σ fw a_j e_j = 0`,
kills the cross term `Σ fw (a_j − ā) m1' e_jᵀ = 0`, leaving exactly

```
Cov_exact(B) − Cov_fit(B) = Σ_j fw_j e_j e_jᵀ = R_m   (PSD).
```

Shifting the intercept, `W0' ← W0' + R_m`, raises `Σ fw S(a_j)` by `R_m` and restores
`Cov_fit(B) = Cov_exact(B)` exactly, while the mean was already exact. So `"mc"`
trades a deliberate pointwise bias (`+R_m` at every `a`) for exact conservation of
the state's total bulk covariance — and that is why `E_S` is measured against the LS
intercept `W0_ls`, not the shifted one.

## Appendix B — why the layer is J·d², not J·d³

(`d = n − 1 ≈ n`; "per bin" = per retained cell `j`.) The premise is right: before
the ReLU each bin needs its own mean and covariance, and materializing a `d×d`
covariance costs `O(d²)` per bin — `J·d²` total is the floor. The question is why
nothing per-bin ever costs `d³`. Three places it could, and what kills each:

**(1) Pushing covariances through the linear map.** A bin's covariance contains
component covariances conjugated by `V`: `S ↦ V S Vᵀ` is `O(d³)`. The binned
companion carries one *free* Gaussian per bin, so it pays one congruence per bin —
`O(B·d³)` per layer. Here the incoming state is the affine family, so every
component's covariance is a **scalar-weighted combination of two fixed matrices**
(three with the atom), and congruence is linear in its matrix argument:

```
V S_k Vᵀ = V (W0 + a_k W1) Vᵀ = G0 + a_k G1        (+ atom: G_at = V S_at Vᵀ)
```

— the `d³` work happens **once per layer on the basis**, before any bin exists.
The hoisting survives all the downstream per-bin algebra because conditioning and
mixture-merging are also linear with scalar weights in the covariance argument, and
every mean-type vector lives in the fixed 7-span `B = [c0|c1|mC_at|g0|g1|g_at|u]`:

```
S_k→j = (G0 + a_k G1 + t2_k uuᵀ) + h_kj g_k g_kᵀ ,      g_k = g0 + a_k g1 + γt2_k u
Ŝ_j   = Σ_k η_k|j [ S_k→j + (m_k→j − m̂_j)(m_k→j − m̂_j)ᵀ ]
      = α_j G0 + β_j G1 + ω_j G_at + B C_j Bᵀ .
```

Everything bin-specific is scalar coordinates (`α_j, β_j, ω_j`, `C_j ∈ ℝ^{7×7}`),
accumulated with `O(K·7²)` scalars per bin (this also removes the *other* naive
`O(K·d²)`-per-bin term — the `K` between-mean outer products per bin are 7×7 sums
instead, `B C_j Bᵀ` once). Materializing `Ŝ_j` costs 3 scaled `d×d` adds plus
`(d×7)(7×7)(7×d)` = `O(d²)` — the same order as writing the matrix down at all.
Ledger: `2–3·d³ (fixed) + J·d²` instead of `J·d³`.

**(2) Factorizations at the ReLU input.** An indefinite `Ŝ_j` would force a per-bin
`eigh` — `O(J·d³)`. It cannot happen: `Ŝ_j` is the total covariance of a genuine
conditional mixture (step 5 merges an actual distribution's moments), hence PSD by
construction; only an `O(d)` diagonal roundoff floor is applied. (Contrast
`fit="pre"`, whose ReLU inputs come from an already-projected family `Σ0 + yΣ1`
that *can* go indefinite — and even there convexity in `y` reduces the certificate
to 2 endpoint Choleskys, not `J`.) The kernel itself is `c_sf·d²` per bin because it
is **entrywise** special functions over the `d×d` covariance — pairwise `Φ₂`
evaluations, never matrix algebra.

**(3) Combining the bins into the fit.** "Fit two `d×d` matrices to `J` matrices"
sounds like heavy linear algebra, but appendix A shows the LS problem decouples into
`d²` independent scalar regressions sharing one 2×2 normal matrix: entry `(a,b)` of
the fit sees only entry `(a,b)` of the data. There is no d-dimensional coupling —
no solve, no inversion, no `d³`. The combine is `Σ_j w_j R_j` and `Σ_j w_j a_j R_j`:
`J` matrix *additions* (`J·d²`), streamed inside the kernel loop so they also cost
no memory; finishing the fit from `(AR, AyR, ā, v_a)` is `O(d²)`, plus `O(J⁺·d²)`
rank-1 updates for `R_m`.

**Total:** `Θ(J·c_sf·d²) + Θ(d³)` per layer, the `d³` fixed (budget-independent,
small constant — it matches the kernel term only when `d ≳ J·c_sf/2`, far beyond
these widths). Every per-bin object is built by *evaluating scalars against
layer-fixed `d×d` matrices*, never by transforming matrices per bin: that is the
entire trick, and it is bought by the state being affine in `a` at both ends of the
layer.

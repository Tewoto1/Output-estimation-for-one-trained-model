# analytic_kprop — runtime analysis of both variants (fit="pre" vs fit="post")

Measured 2026-07-14 on a 4-core / 3.9 GB sandbox VM (numpy + OpenBLAS, scipy special),
depth 2, `num_nodes = 40`, `bulk_relu = "exact"`, W2 grid, coordinate-spike nets
`M = W + e₁e₁ᵀ`. Per-phase timers are always on: every run logs
`stats["t_{params,grid,pairs,cells,fit,diag,relu,merge}"]` (seconds per layer), so
these tables can be regenerated on any machine from `collect=True` output.

---

## 1. The parts, in execution order

Both variants run the same five-stage cycle per hidden layer; they differ in **where
the affine projection happens** and therefore in what the state carries.

### fit="pre" (paper Algorithm 7.2) — state: `{p_i, a_i, m_i, S_i}`, an `(m, d, d)` stack

| phase | what it does | cost | @n=1024 |
|---|---|---|---|
| `params` | component params under the layer blocks (eqs 60–61): `S_i r` einsums, `m_Y`, `s²_Y`, `m_C`, `g` per node | `O(m d²)` | 0.01 s |
| `grid` | negative-mass split + Lloyd–Max W2 quantization of the exact scalar mixture (vectorized `_mixture_cells_vec`) | `O(iters·m·J)` scalars | 0.10 s |
| `pairs` | truncated-normal cell stats `Q, δ, v` for every (component, cell), one `(m, J)` broadcast (eqs 63–66) | `O(mJ)` special fns | <0.01 s |
| `cells` | cell masses/centroids + merged conditional means `m̂_j` (eqs 69–71) | `O(mJd)` | <0.01 s |
| `fit` | the affine re-projection (eqs 86–87, 90): grid-weighted covariance sums `T0, T1` via **2 aggregated congruences** `V(ΣᵢcᵢSᵢ)Vᵀ` + `O(m d²)` reductions, then LS + `R_m` | `O(d³)+O(m d²)` | 0.15 s |
| `relu` | per retained node: affine slice `Σ₀+y_jΣ₁` (endpoint-Cholesky PSD screen, 2/layer) → **exact bivariate Gaussian-ReLU kernel** → `(r_j, R_j)` stored into a `(J, d, d)` stack; threaded (`workers`) | `O(J d²)` special fns + `O(J d²)` memory writes | **7.80 s** |
| `merge` | zero-atom total-expectation/covariance merge (eqs 40–42) + assembling the next `(m, d, d)` state stack | `O(J d²)` | 0.15 s |

Peak extra RAM ≈ `3.5 · num_nodes · d² · 8 B` (the `R` stack + next state + merge
copies): **0.8 GB at n=1024, OOM-killed at n=2048 on this box.**

### fit="post" — state: `(p_k, a_k)` + `(m0, m1, W0, W1)` [+ exact atom], `O(d²)` total

| phase | what it does | cost | @n=1024 |
|---|---|---|---|
| `params` | **the whole linear step**: `c0 = Vm0+η`, `c1 = u+Vm1`, `g0 = VW0r`, `g1 = VW1r`, `s0/s1 = rᵀW·r`, `G0 = VW0Vᵀ`, `G1 = VW1Vᵀ` (+ `G_at` for the exact atom) — 2–3 congruences + 4 matvecs, **independent of num_nodes** | `O(d³)` fixed | 0.13 s |
| `grid` | same as pre | `O(iters·m·J)` | 0.11 s |
| `pairs` | same as pre | `O(mJ)` | <0.01 s |
| `cells` | everything is affine in `a_i`, so all cell-merged moments live in the ≤7-vector basis `[c0, c1, mC_at, g0, g1, g_at, u]`: coefficient tensors `(J, 7, 7)` + `m̂_j` | `O(mJ·49)+O(Jd)` | <0.01 s |
| `relu` | per node: **dense assembly from the factored form** `αⱼG0+βⱼG1+ωⱼG_at+B CⱼBᵀ` (`O(d²)`; mixture ⇒ PSD by construction, diagonal floor only) → same exact kernel → **streaming slot accumulators** `ΣwR, ΣwyR, ΣwR_neg` (no `(J,d,d)` storage) | `O(J d²)` special fns + assembly | **4.58 s** |
| `merge` | zero-atom merge from stored `r_j` rows + `ΣwR_neg` | `O(J_neg d)+O(d²)` | <0.01 s |
| `fit` | post-ReLU weighted LS of `(m0, m1)` on `r_j` and `(W0, W1)` from the two accumulators (+`R_m` mc-intercept); atom kept exact (`atom="exact"`) or included as the `a=0` data point (`atom="fit"`) | `O(Jd)+O(d²)` | 0.03 s |

Peak extra RAM ≈ `(n_slots·3 + kernel temporaries) · d² · 8 B`: **~0.1 GB at n=1024;
n=2048 runs in 18.2 s where pre cannot run at all.**

---

## 2. Measured width sweep (nn=40, workers=auto, totals in seconds)

| n | pre total | pre relu | pre other | post total | post relu | post other | pre peak RAM |
|---|---|---|---|---|---|---|---|
| 128 | 0.30 | 0.21 | 0.09 | 0.21 | 0.12 | 0.09 | ~0 |
| 256 | 0.67 | 0.57 | 0.10 | 0.31 | 0.22 | 0.09 | +0.11 GB |
| 512 | 2.58 | 2.42 | 0.16 | 1.76 | 1.65 | 0.11 | +0.34 GB |
| 1024 | 8.21 | 7.80 | 0.41 | 4.85 | 4.58 | 0.27 | +0.80 GB |
| 2048 | **OOM** | — | — | 18.2 | ~17.5 | ~0.6 | (post: ~+2 GB) |

The ReLU kernel is **93–96% of runtime in both variants at every width**; the entire
propagation machinery (linear step, grid, reconditioning, fit, merge) is 0.1–0.4 s
even at n=1024. "pre other" grows with width (the `(J,d,d)` stack traffic shows up in
`fit`+`merge`); "post other" is flat apart from the fixed 2–3 congruences in `params`.
Post's kernel-phase advantage (≈1.7× at n≥512) is **memory traffic**, not arithmetic:
pre writes a 0.7 GB `R` stack and re-reads it in `merge`; post accumulates into
`n_slots` small partials.

## 3. Scaling in the node budget (n=512, `t_relu`, workers=auto)

| num_nodes | 10 | 20 | 40 | 80 |
|---|---|---|---|---|
| pre | 0.31 | 0.45 | 2.42 | 4.19 |
| post | 0.43 | 0.76 | 1.65 | 2.90 |

Linear in `num_nodes` once past fixed overheads — the kernel is called once per
retained cell. Combined with the accuracy knee at ~6–10 nodes (§3 of the notebook /
the scaling script), `num_nodes ≈ 10–20` buys nearly all the accuracy at a quarter of
the cost. (At tiny budgets post's fixed per-node assembly makes it slightly slower
than pre; it wins from nn≈40 up.)

## 4. Inside the kernel (n=512, serial, 80 calls, `_utils.exact_relu_covariance`)

| piece | pre | post | share |
|---|---|---|---|
| `bvn_cdf` (Owen's T bivariate CDF) | 1.58 s | 1.44 s | ~60% |
| kernel bookkeeping (rho, masks, Ecross algebra) | 0.45 s | 0.34 s | ~15% |
| `Φ`/`φ` evaluations | 0.34 s | 0.32 s | ~13% |
| `_a` + `_bvn_pdf` | 0.37 s | 0.23 s | ~10% |

So ~60% of *total program time* is `scipy.special.owens_t`. Any further speedup must
attack this kernel: upper-triangle-only evaluation (`Ecross` is symmetric, ~1.7×),
the cheap `bulk_relu="gain"` backend (`O(d²)` elementary ops, no bvn), or fewer nodes.

## 5. Serial vs threaded (n=512, nn=40)

pre 2.54 s → 2.4–2.6 s; post 2.55 s → 1.65–1.76 s on this 4-core, bandwidth-starved
VM (scipy ufuncs release the GIL but the work is memory-bound; expect better scaling
on real host CPUs). Pre threading is bit-identical to serial (disjoint writes); post
is `allclose` (~1e-16) — slot-grouped accumulation regroups floating-point sums.

## 6. Asymptotic summary (per layer; m ≈ J ≈ num_nodes, bulk dim d)

| | fit="pre" | fit="post" |
|---|---|---|
| linear + recondition + fit | `O(d³) + O(m d²)` | `O(d³)` fixed + `O(mJ)` scalars |
| ReLU step | `O(J d²)` special fns | `O(J d²)` special fns + `O(J d²)` assembly |
| state memory | `O(J d²)` | `O(d²)` |
| PSD handling | 2 Cholesky/layer (affine convexity) | none needed (mixture ⇒ PSD) |
| extra approximation | affine fit of the reconditioned pre-activation | affine fit of the nonlinear post-ReLU `r(a), R(a)` (paper checklist-7's excluded projection) + optional atom-in-fit |

Accuracy at matched budget is at parity so far (n=48: post 9.5e-3 vs pre 1.0e-2;
depth-1 post is pure quadrature, 4e-8 at 40 nodes); the A100 scaling run
(`--fit both`) is the decisive test.

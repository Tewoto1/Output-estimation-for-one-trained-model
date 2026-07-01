# Coordinate-spike binned K-propagation — full algorithm, algebra, and Wasserstein bin placement

This is the complete derivation behind `Mecha_preds.binned_kprop` (the **K = 2** predictor of
`E[model(X)]`, `X ~ N(0, I_n)`, for a ReLU MLP whose hidden matrices carry a single-coordinate
spike `M = W + e₁e₁ᵀ`). It walks the state, every algebra step of the linear and ReLU updates,
and how the **bin points are chosen by minimizing the Wasserstein-2 distance to the expected
continuous distribution of the spike coordinate**.

Notation: width `n`, spike axis `e = e₁` (coordinate `0`), bulk dimension `d = n − 1`, number of
bins `m = num_bins`, cumulant order `K = 2`. `φ`, `Φ` are the standard-normal pdf/cdf.

---

## 1. Setup: spike/bulk split

Every layer vector is split along the spike axis,

$$X = A\,e + B,\qquad A = e^\top X\in\mathbb R,\quad B = \Pi X\in e^\perp,\ \Pi = I-ee^\top .$$

`A` is the scalar **spike coordinate**; `B` is the `d`-dim **bulk**. A hidden matrix block-decomposes
relative to `ℝⁿ = ℝe ⊕ e^⊥` as

$$M=\begin{pmatrix}\gamma & r^\top\\ u & V\end{pmatrix},\quad
\gamma=M_{00},\ r=M_{0,1:},\ u=M_{1:,0},\ V=M_{1:,1:},$$

so one linear layer acts as

$$\boxed{\,A^+=\gamma A + r^\top B,\qquad B^+ = uA + VB.\,}\tag{1}$$

**Why bin the spike.** For a *flat* spike `(1/n)\mathbf 1\mathbf 1^\top` the spike direction gets a
flat-loop `1/n` discount and ordinary total-order kprop is accurate. A *coordinate* spike has no such
discount: (1) preserves a large residue of coordinate 0, so the cumulants of `A` are `O(1)` at **every**
order and cannot be truncated at `K = 2`. We therefore represent `A` **non-parametrically** by a
discrete law over `m` bins (a hidden-Markov model over the spike coordinate), and propagate the bulk
`B` by ordinary `K = 2` cumulant propagation **conditional on each bin**.

---

## 2. State

The joint law of `X` is approximated by a **mixture**

$$\widehat{\mathcal L}(X)=\sum_{\alpha=1}^{m} p_\alpha\,\delta_{v_\alpha}(dA)\otimes\nu_\alpha(dB),
\qquad \nu_\alpha\approx\mathcal N(\mu_\alpha,\Sigma_\alpha),$$

stored (class `BinnedK2State`) as

| symbol | shape | meaning |
|---|---|---|
| `p[α]` | `(m,)` | `P(A ∈ bin α)`, `≥0`, sums to 1 |
| `a[α] = v_α` | `(m,)` | representative spike value `≈ E[A | bin α]` |
| `mu[α] = μ_α` | `(m,d)` | conditional bulk mean `E[B | bin α]` |
| `Sigma[α] = Σ_α` | `(m,d,d)` | conditional bulk covariance `Cov[B | bin α]` |

**Rule:** coordinate 0 never appears inside `μ`/`Σ`; the spike lives only in `(p, a)`. Within a bin, `A`
is collapsed to the single value `v_α` (a point mass) — this is the one approximation the binning
introduces in the spike direction, and §7 chooses the bins to make it as small as possible.

---

## 3. Initialization (`X⁰ ~ N(0, I)`)

`A⁰ ~ N(0,1) ⟂ B⁰ ~ N(0, I_d)`. With pre-activation edges `z₀=−∞ < z₁ < … < z_m=+∞`, bin `α=[z_{α−1},z_α)`:

$$p_\alpha=\Phi(z_\alpha)-\Phi(z_{\alpha-1}),\qquad
v_\alpha=E[A^0\mid A^0\in\text{bin }\alpha]=\frac{\varphi(z_{\alpha-1})-\varphi(z_\alpha)}{\Phi(z_\alpha)-\Phi(z_{\alpha-1})},$$

$$\mu_\alpha=0,\qquad \Sigma_\alpha=I_d .$$

(`v_α` is the truncated-normal mean — already the Wasserstein-optimal representative, §7.)

---

## 4. Linear step — algebra

Inside old bin `α` we have `A = v_α` (a point) and `B ~ N(μ_α, Σ_α)`. Push both through (1).

### 4.1 The joint Gaussian `(Y, C)`
Define the new **scalar** spike pre-activation `Y_α` and the new **bulk** vector `C_α`:

$$Y_\alpha=\gamma v_\alpha + r^\top B,\qquad C_\alpha=u v_\alpha + V B .$$

Both are affine in the Gaussian `B`, hence jointly Gaussian with

$$m_Y=\gamma v_\alpha + r^\top\mu_\alpha,\quad s_Y^2=r^\top\Sigma_\alpha r,\qquad
m_C=u v_\alpha + V\mu_\alpha,\quad \Sigma_C=V\Sigma_\alpha V^\top,$$
$$g:=\mathrm{Cov}(C_\alpha,Y_\alpha)=V\Sigma_\alpha r .\tag{2}$$

### 4.2 Transition kernel (bin → bin)
For each new bin `I_β=[ℓ_β,h_β)`, with standardized limits `a=(ℓ_β−m_Y)/s_Y`, `b=(h_β−m_Y)/s_Y`:

$$Q_{\beta\alpha}=P(Y_\alpha\in I_\beta)=\Phi(b)-\Phi(a).$$

New bin masses and the **Bayes posterior** of the old bin given the new one:

$$p^+_\beta=\sum_\alpha p_\alpha Q_{\beta\alpha}\quad(\;p^+ = Q\,p\;),\qquad
\eta_{\alpha\mid\beta}=\frac{p_\alpha Q_{\beta\alpha}}{p^+_\beta}.\tag{3}$$

### 4.3 Conditioning the bulk on `Y ∈ I_β`
Truncated-normal moments of `Y` on `I_β` (`Z = Q_{\beta\alpha}`):

$$\tau_1=E[Y-m_Y\mid I_\beta]=s_Y\frac{\varphi(a)-\varphi(b)}{Z},\quad
\tau_2=E[(Y-m_Y)^2\mid I_\beta]=s_Y^2\Big[1+\frac{a\varphi(a)-b\varphi(b)}{Z}\Big],$$
$$\mathrm{Var}(Y\mid I_\beta)=\tau_2-\tau_1^2 .$$

Use the Gaussian regression of `C` on `Y` (exact because `(Y,C)` is jointly Gaussian):

$$C_\alpha = m_C + \frac{g}{s_Y^2}\,(Y_\alpha-m_Y) + C_\perp,\qquad C_\perp\perp Y_\alpha,\ \
\mathrm{Cov}(C_\perp)=\Sigma_C-\frac{gg^\top}{s_Y^2}.$$

Taking conditional mean/cov over `Y ∈ I_β` (only the `Y`-dependent part feels the truncation):

$$\boxed{\;\mu_{\alpha\to\beta}=m_C+\frac{g}{s_Y^2}\,\tau_1,\qquad
\Sigma_{\alpha\to\beta}=\underbrace{\Sigma_C-\frac{gg^\top}{s_Y^2}}_{\text{cov given exact }Y}
+\underbrace{\frac{gg^\top}{s_Y^4}\,(\tau_2-\tau_1^2)}_{\text{re-add }Y\text{'s in-bin spread}}.\;}\tag{4}$$

(The degenerate `s_Y^2≈0` branch: `Y_α` is the constant `m_Y`, so it lands wholly in the bin containing
`m_Y` with `μ_{α→β}=m_C`, `Σ_{α→β}=Σ_C`, no division by `s_Y^2`.)

### 4.4 Mixture over old bins (law of total mean / covariance)
Several old bins feed new bin `β`; combine them with the posterior weights (3):

$$v^+_\beta=\sum_\alpha\eta_{\alpha\mid\beta}\,(m_{Y,\alpha}+\tau_{1,\alpha\beta})
=\sum_\alpha\eta_{\alpha\mid\beta}\,E[Y_\alpha\mid I_\beta],$$
$$\mu^+_\beta=\sum_\alpha\eta_{\alpha\mid\beta}\,\mu_{\alpha\to\beta},\qquad
\Sigma^+_\beta=\sum_\alpha\eta_{\alpha\mid\beta}\Big[\Sigma_{\alpha\to\beta}
+(\mu_{\alpha\to\beta}-\mu^+_\beta)(\mu_{\alpha\to\beta}-\mu^+_\beta)^\top\Big].\tag{5}$$

The last term is the between-component spread — the law of total covariance. The representative `v⁺_β`
is the **dynamic** conditional mean `E[A⁺ | A⁺∈I_β]`, not a fixed bin center. Finally renormalize
`p⁺ ← p⁺/∑p⁺` (logging the tail mass lost) and symmetrize/PSD-clip `Σ⁺` (logging any clip).

**Cost.** The only `O(n³)` work is `Σ_C=VΣ_αV^\top` per old bin; everything else is `O(m^2 d^2)`. The
implementation never forms the spec's `(m,m,d,d)` tensor — it stores the `β`-independent
`Σ_C−gg^\top/s_Y^2` once per `α` and re-adds the rank-1 `gg^\top` term per new bin.

---

## 5. ReLU step — algebra

ReLU is coordinatewise, so it splits: `ρ(X) = ρ(A)\,e + ρ(B)`.

### 5.1 Bulk, inside each bin (exact `K=2`)
With `B ~ N(μ_α,Σ_α)`, `σ_i²=Σ_{α,ii}`, `z_i=μ_i/σ_i`:

$$E[\rho(B_i)]=\mu_i\Phi(z_i)+\sigma_i\varphi(z_i),\qquad
E[\rho(B_i)^2]=(\mu_i^2+\sigma_i^2)\Phi(z_i)+\mu_i\sigma_i\varphi(z_i),$$

giving `μ̃_{α,i}=E[ρ(B_i)]`, `Σ̃_{α,ii}=E[ρ(B_i)^2]−μ̃_{α,i}^2`. Off-diagonal (default backend `"exact"`)
is the **exact bivariate-Gaussian** covariance `Cov(ρ(B_i),ρ(B_j))` via Owen's-T
(`_utils.exact_relu_covariance`); the cheaper `"gain"` backend uses the leading-order
`Σ̃_{ij}=Φ(z_i)Φ(z_j)\,Σ_{ij}`; `"kprop"` delegates to the ordinary harmonic kprop ReLU.

### 5.2 Spike + merge
ReLU output is nonnegative, so the spike representatives map `ṽ_α=\max(v_α,0)` onto a **nonnegative**
post-grid with a zero bin. Old bins landing in the same post bin `β`, `S_β=\{\alpha:\text{bin}(ṽ_α)=\beta\}`,
merge by mixture moments (`η_{α|β}=p_α/p^+_β`):

$$p^+_\beta=\sum_{\alpha\in S_\beta}p_\alpha,\quad
v^+_\beta=\sum_{\alpha\in S_\beta}\eta_{\alpha\mid\beta}\,ṽ_\alpha,\quad
\mu^+_\beta=\sum\eta_{\alpha\mid\beta}\,\tilde\mu_\alpha,\quad
\Sigma^+_\beta=\sum\eta_{\alpha\mid\beta}\big[\tilde\Sigma_\alpha+(\tilde\mu_\alpha-\mu^+_\beta)(\cdot)^\top\big].$$

Typically *all* bins with `v_α≤0` collapse into the zero bin — that is the post-ReLU mass atom at 0 (§7.4).

---

## 6. Readout and recovering moments

A full network alternates `linear_step` (pre-edges) and `relu_step` (post-edges) over the hidden layers;
the readout is linear (no ReLU). After the last hidden ReLU, reconstruct the full mean

$$E[X]=\bar A\,e+\bar B,\qquad \bar A=\sum_\alpha p_\alpha v_\alpha,\quad \bar B=\sum_\alpha p_\alpha\mu_\alpha,$$

and the output mean is `W_ro E[X] + b_ro`. (Full covariance, if needed, by total covariance over the
finite spike law — `unconditional_mean_cov`.)

---

## 7. Determining the bin points by Wasserstein distance

### 7.1 The right error to minimize
Binning replaces the continuous law of the scalar `A` by the `m`-atom law
`Â=∑_α p_α δ_{v_α}`. Within a cell, every `A` is reported as `v_α`, so the **squared error the binning
introduces in the spike direction** is

$$\sum_\alpha\int_{\text{cell }\alpha}(a-v_\alpha)^2\,f(a)\,da
=\sum_\alpha p_\alpha\,\mathrm{Var}(A\mid\text{cell }\alpha)
= W_2^2\big(\mathcal L(A),\ \widehat{\mathcal L}(A)\big).\tag{6}$$

This **is** the squared Wasserstein-2 distance between the true continuous `A` and its binned version
(the nearest-representative coupling is the optimal one). So "choose the bins well" = "minimize `W₂` to
the expected continuous distribution of `A`," and (6) is exactly the in-bin variance that the linear
step's representative-collapse throws away. Minimizing `W₂` directly minimizes the algorithm's
spike-discretization error.

### 7.2 Optimality conditions (Lloyd–Max)
Minimizing (6) over both the partition and the representatives gives the two stationarity conditions of
optimal scalar quantization:

1. **Representatives = cell centroids:** `v_α = E[A | cell α]` (∂/∂v_α = 0).
2. **Edges = midpoints between adjacent representatives:** `z_α = ½(v_α + v_{α+1})` (nearest-representative
   partition).

Condition 1 is already what the algorithm uses everywhere (the truncated-normal / conditional means in
§3–§5). Condition 2 is the part a fixed **equal-mass quantile** grid does *not* satisfy. Alternating the
two is **Lloyd's algorithm**; for a log-concave `f` (Gaussian, rectified Gaussian) it converges to the
global optimum.

```
Lloyd–Max(f, m):
    init edges  z  (e.g. equal-mass quantiles of f)
    repeat:
        v_α ← E_f[A | z_{α-1} ≤ A < z_α]        # centroids
        z_α ← ½ (v_α + v_{α+1})                  # midpoints
    until edges converge
```

### 7.3 The expected continuous distribution per layer
At each layer the closure gives `A`'s expected continuous law in closed form, and we quantize *that*:

- **Pre-activation grid** (before a linear step): `A⁺` is, under the `K=2` closure, **exactly a Gaussian
  mixture** — one component per current bin,
  $$\mathcal L(A^+)=\sum_\alpha p_\alpha\,\mathcal N\!\big(m_{Y,\alpha},\,s_{Y,\alpha}^2\big),\qquad
  m_{Y,\alpha}=\gamma v_\alpha+r^\top\mu_\alpha,\ \ s_{Y,\alpha}^2=r^\top\Sigma_\alpha r .$$
  Includes negatives (`A⁺` can be `<0` even when the incoming `A≥0`).
  **The cell centroid under a mixture is closed form** (no quadrature): with
  `α_k=(a−m_k)/s_k`, `β_k=(b−m_k)/s_k`,
  $$E[A^+\mid[a,b)]=\frac{\sum_k w_k\big[m_k(\Phi(\beta_k)-\Phi(\alpha_k))+s_k(\varphi(\alpha_k)-\varphi(\beta_k))\big]}
  {\sum_k w_k(\Phi(\beta_k)-\Phi(\alpha_k))}.$$
  This is *exactly* the linear step's representative `v^+_\beta=\sum_\alpha\eta_{\alpha\mid\beta}E[Y_\alpha\mid I_\beta]`
  (since `η_{α|β}=p_α Q_{βα}/p^+_β` and `Q_{βα}` is component `α`'s mass in the cell). So in the
  propagation the per-bin expected value is the **exact mixture centroid**, never a single-Gaussian
  approximation.
- **Post-ReLU grid:** `A_{\text{post}}=\max(A_{\text{pre}},0)` is the **rectified mixture** — an atom at 0
  of mass `∑_k w_k Φ(−m_k/s_k)` plus a positive truncated mixture tail. Its `W₂`-optimal quantizer puts
  **one representative on the 0-atom** and Lloyd–Max points on the positive tail.

Two grid builders: `lloyd_max_edges(mean, std, num_bins, rectified=…)` Lloyd-Maxes a single
moment-matched Gaussian (cheap, unimodal), while `lloyd_max_edges_mixture(weights, means, stds,
num_bins, rectified=…)` Lloyd-Maxes the **true mixture** using the closed-form centroid above — every
iteration closed form, no quadrature. On a 3-component test the mixture version cut `W₂²` to the true
law by 14–28% vs the moment-matched one.

### 7.4 Why `W₂`-optimal beats equal-mass quantiles
Equal-mass quantiles put point density `∝ f`. The `W₂` (squared-error) optimum puts point density
`∝ f^{1/3}` (Panter–Dite / high-rate quantization), i.e. **more resolution in the tails** — exactly where
the amplified spike coordinate carries the mass that ReLU treats asymmetrically. Measured `W₂²`
distortion of `N(0,1)` (this repo, `lloyd_max_edges` vs `make_gaussian_edges`):

| m | equal-mass | Lloyd–Max (`W₂`) | reduction | extreme reps (equal-mass → `W₂`) |
|---|---|---|---|---|
| 5  | 1.03e-1 | 7.99e-2 | 22% | |
| 9  | 4.70e-2 | 2.79e-2 | 41% | ±1.70 → ±2.25 |
| 15 | 2.42e-2 | 1.07e-2 | 56% | ±… → ±2.68 |
| 31 | 9.59e-3 | 2.70e-3 | 72% | ±… → ±3.17 |

Both decay as `W₂² ~ C/m²` (so the spike-discretization RMS error `~ 1/m`); Lloyd–Max has the smaller
constant `C`, and the gap widens with `m`.

### 7.5 Relation to the implementation
- **Representatives** are already `W₂`-optimal everywhere (conditional means in §3–§5).
- **Default edges** are equal-mass quantiles (`make_gaussian_edges` / `make_relu_post_edges`) — simple and
  guaranteed non-empty (good for the transition kernel), but not `W₂`-optimal.
- **`W₂`-optimal edges** are available via `lloyd_max_edges` (single moment-matched Gaussian) and
  `lloyd_max_edges_mixture` (the exact mixture, closed-form centroids) — both: centroid reps + midpoint
  edges, with the 0-atom pinned for the rectified/post case. Verified: interior edges equal the midpoints
  of adjacent reps, the single-Gaussian distortions match the table above, and the mixture version is
  exact for the true `A⁺` law.
- **Note** these are grid *utilities*; the propagation loop currently still builds equal-mass quantile
  grids once and reuses them. Wiring a per-layer `lloyd_max_edges_mixture` re-grid (the spike mixture is
  rebuilt from `(p, m_Y, s_Y)` each layer) is the principled "dynamic `W₂` binning" — and since the
  representatives are *already* exact mixture centroids, only the edges would change.

A practical caveat from the width experiments: past ≈ 4–7 bins the predictor's accuracy is limited by the
**bulk `K=2` Gaussian closure**, not by the spike discretization, so `W₂`-optimal binning mainly lets you
reach the same floor with **fewer** bins (most useful in the few-bin / deeper-net regime). To push *below*
the floor needs higher bulk cumulant order, not finer spike bins.

---

## 8. Error sources (summary)

1. **Spike discretization** — within-bin collapse of `A`; equals `W₂²(\mathcal L(A),\widehat{\mathcal L}(A))`
   (6); `~ 1/m²`, minimized by §7; **and you need ≥1 negative bin** (a single `[-∞,0)` catch-bin suffices —
   dropping negatives entirely is a 30–60% error).
2. **Bulk `K=2` Gaussian closure** — `B | bin` treated as Gaussian; the dominant, width-dependent floor
   (`MSE ~ n^{-2}`, the budget rate).
3. **Mixture re-Gaussianization** — refitting `(5)`’s mixture as one Gaussian per bin.

Numerically: tail mass loss before renormalization and PSD clips are logged; both are `~10^{-16}` in
practice.

---

## Appendix — closed forms

Standard normal `φ(x)=e^{-x²/2}/√{2π}`, `Φ=` its CDF (scipy `ndtr`).

Truncated normal on `[a,b]`, `Z=Φ(b)−Φ(a)` (with `xφ(x)→0` as `x→±∞`):

$$E[A\mid a≤A<b]=\frac{φ(a)−φ(b)}{Z},\qquad
\mathrm{Var}=1+\frac{aφ(a)−bφ(b)}{Z}-\Big(\frac{φ(a)−φ(b)}{Z}\Big)^2 .$$

Rectified Gaussian moments `Y~N(μ,σ²)`, `z=μ/σ`:
`E[ρ(Y)]=μΦ(z)+σφ(z)`, `E[ρ(Y)²]=(μ²+σ²)Φ(z)+μσφ(z)`, atom mass `P(Y≤0)=Φ(−z)`.

All implemented torch-free in `Mecha_preds._utils` (canonical shared kernel) and used by `core.py`.

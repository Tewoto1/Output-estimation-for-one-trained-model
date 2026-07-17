# Transverse cumulants through the spiked linear layer: exact size characterization

*2026-07-17. Companion verification: `experiments/transverse_pass_characterization/verify_transverse_pass.py` (exact identities MC-checked; model-level scalings measured on the e1-spiked ReLU net). Notation matches `Mecha_preds/cumulants/spikekprop/core.py` and `experiments/e1_cumulant_scaling`.*

---

## 0. Result

**Setting.** $a \in \mathbb{R}^n$ is the post-ReLU activation at a hidden layer; the next layer is $h = Ma$ with $M = W' + \theta v v^\top$, $W'_{ij} \sim N(0, 1/n)$ iid and independent of $a$, $v$ unit ($v=e_1$ here), $|\theta| = O(1)$. A *transverse output direction* is a fixed unit $u \perp v$ (in practice an output coordinate $e_b$, $b \neq 1$). Write $T_r := \kappa^{(r)}(a)$ for the order-$r$ connected cumulant tensor, $T_r^{(p)}$ for the order-$(r-2p)$ tensor obtained by tracing $p$ disjoint slot-pairs ($T^{(p)}_{i_{2p+1}\cdots i_r} = \sum_{j_1\cdots j_p} T_{j_1 j_1 \cdots j_p j_p\, i_{2p+1}\cdots i_r}$; by symmetry only $p$ matters), and $\mathrm{pt}_r := T_r^{(r/2)}$ for the full pair trace (even $r$).

**(R1) The layer is exactly a Gaussian contraction of the input cumulant tensor.** Since $u^\top M = u^\top W'$ (the spike drops out exactly for $u \perp v$), multilinearity of joint cumulants gives, for every $r \ge 1$,

$$\kappa_r\big(u \cdot h \,\big|\, W'\big) \;=\; T_r[w^{\otimes r}], \qquad w := W'^\top u \sim N(0, I_n/n) \ \text{exactly.}$$

So the question "what does the linear layer pass?" reduces to: *which tensors $T_r$ have small contractions against an iid Gaussian probe.* Distinct orthonormal $u$'s give independent $w$'s.

**(R2) The complete, minimal functional family (per order $r$).** Over the weight ensemble,

$$\mathbb{E}_{W'}\, T_r[w^{\otimes r}] = \frac{(r-1)!!}{n^{r/2}}\,\mathrm{pt}_r \ \ (r \text{ even; } 0 \text{ odd}), \qquad
\mathrm{Var}_{W'}\, T_r[w^{\otimes r}] = \frac{1}{n^{r}} \sum_{p=0}^{\lceil r/2\rceil - 1} a_{r,p}\, \big\|T_r^{(p)}\big\|_F^2,$$

$$a_{r,p} = \left[\binom{r}{2p}(2p-1)!!\right]^2 (r-2p)! \;>\; 0.$$

| $r$ | mean coeff $(r{-}1)!!$ | $a_{r,0}$ | $a_{r,1}$ | $a_{r,2}$ |
|---|---|---|---|---|
| 2 | 1 | 2 | — | — |
| 3 | 0 | 6 | 9 | — |
| 4 | 3 | 24 | 72 | — |
| 5 | 0 | 120 | 600 | 225 |
| 6 | 15 | 720 | 5400 | 4050 |

Because every $a_{r,p} > 0$, **no cancellation between functionals is possible**: the family $\{\mathrm{pt}_r\} \cup \{\|T_r^{(p)}\|_F : p < r/2\}$ is both sufficient *and necessary*. Boxed, per order:

$$\kappa_r(u\cdot h) = O\big(n^{1-r/2}\big)\ \text{in mean and RMS over seeds/directions} \iff \mathrm{pt}_r = O(n)\ \text{and}\ \|T_r^{(p)}\|_F = O(n)\ \forall p < r/2.$$

**(R3) The uniform, all-orders form — "contracted diagrams are $O(n)$", made exact.** Say $a$ lies in the **diagram class** $\mathcal{D}(n)$ if every *closed connected contraction diagram* — take any multiset of connected cumulant tensors $\kappa^{(r_1)},\dots,\kappa^{(r_m)}$ of $a$ (the mean vector $\mu$ allowed as an order-1 vertex), pair up all slots, contract, sum — has value $O(n)$. Then:

- one-vertex diagrams are exactly the $\mathrm{pt}_r$ (systematic part, R2 mean);
- two-vertex diagrams are exactly the $\|T_r^{(p)}\|_F^2$ and mixed-order inner products (so $\|T_r^{(p)}\|_F = O(\sqrt n)$, making scatter $\sqrt n$-subdominant to the mean);
- $m$-vertex diagrams control the $m$-th $W$-moments of $\kappa_r^\perp$ (Gaussian-chaos growth).

**iid coordinates saturate every connected diagram at exactly $\Theta(n)$** (all tensors diagonal, connectivity forces one shared index, the sum contributes one factor of $n$). So $\mathcal{D}(n)$ = "no connected diagram larger than iid". Inside $\mathcal{D}(n)$, for a fixed transverse direction:

| order | systematic | scatter (over seeds/directions) |
|---|---|---|
| $r=1$ | $0$ | $O(1)$ — tracked exactly by mean-prop |
| $r=2$ | $\mathrm{tr}\,\Sigma / n = O(1)$ | $O(n^{-1/2})$ |
| $r$ even $\ge 4$ | $(r{-}1)!!\,\mathrm{pt}_r\, n^{-r/2} \sim n^{1-r/2}$ | $n^{(1-r)/2}$ |
| $r$ odd $\ge 3$ | $0$ | $n^{(1-r)/2}$ |

Note the odd orders: generic directions carry random signs, so odd cumulants cancel to $n^{(1-r)/2}$ — a full $\sqrt n$ *below* the naive $n^{1-r/2}$. (Concretely $\kappa_3^\perp \sim n^{-1}$, so the first neglected term in a $K{=}2$ closure contributes squared error $\sim n^{-2}$ — the verified binned-kprop width law drops out of this counting.)

**(R4) Why the $\|a\|_2^2$ criterion was necessary-but-not-sufficient.** Exact (it is the identity already used in `normsq_cumulant_scaling`): annealed over $W'$,

$$\kappa_{2r}(u\cdot h)^{\mathrm{annealed}} = (2r-1)!!\; \kappa_r\big(\|a\|^2\big)/n^r,$$

because $u\cdot h \mid a \sim N(0, \|a\|^2/n)$ — a Gaussian scale mixture. Moreover (Leonov–Shiryaev) the norm-square cumulants are themselves a diagram aggregate:

$$\kappa_r\big(\|a\|^2\big) = \sum_{\substack{\text{closed connected diagrams } D \\ \text{with } r \text{ contraction edges}}} N_D \cdot \mathrm{val}(D), \qquad N_D \in \mathbb{Z}_{>0}.$$

Hence $\mathcal{D}(n) \Rightarrow \kappa_r(\|a\|^2) = O(n)$ — your criterion is the **annealed shadow** of the diagram class — but *not conversely*, for three separately fatal reasons:

1. **Quenching.** The network has one $W$. The annealed cumulant resums quenched fluctuations (e.g. $\kappa_4^{\mathrm{ann}} = \mathbb{E}_W \kappa_4^{(W)} + 3\,\mathrm{Var}_W \kappa_2^{(W)}$); the aggregate mixes *signed* diagram values, so individual diagrams can be large and cancel in $\kappa_r(\|a\|^2)$ while different $W$-moments weight them with *different positive* combinations (R2).
2. **Odd blindness.** A centered Gaussian scale mixture is symmetric: the annealed law carries *zero* information at odd orders. Quenched $\kappa_{\mathrm{odd}}^\perp$ has mean $0$ but scatter $n^{(1-r)/2}$ governed entirely by the odd-tensor functionals — invisible to any $\|a\|^2$ statistic.
3. **Direction blindness.** An $O(1)$-Frobenius low-rank tensor (see R5) shifts an $O(n)$ aggregate by $O(1)$ — undetectable — yet is $O(1)$ along its own direction.

So the correct completion of "assume $\kappa_r(\|a\|^2)$ small" is: assume the *individual* connected diagrams small. That is exactly the content of "connected cumulants generalize this and are enough".

**(R5) The spike refinement and the closed loop.** The induction-stable class for the $e_1$-spiked network is *conditional*:

$$\mathcal{C}:\qquad \kappa^{(r)}\big(a_\perp \,\big|\, S\big) \in \mathcal{D}(n) \ \text{uniformly in } S \qquad \oplus \qquad \text{an } S\text{-coherent low-rank part},$$

with $S = v\cdot a$ the spike component. By the law of total cumulance (Brillinger), unconditional tensors gain terms built from cumulants of the conditional-cumulant *functions* of $S$; with the (verified, layer-2) rank-1 conditional-mean knee $m(S) \approx \text{affine} + g(S)\,u_c$, $\|u_c\| = O(1)$, entries $O(n^{-1/2})$, these anomaly tensors are $\kappa_j(g)\, u_c^{\otimes j} \otimes \cdots$: **all their traces and Frobenius norms are $O(1)$** — inside every R2 budget, and harmless through fresh generic directions (each leg closes at $\langle u_c, w\rangle \sim n^{-1/2}$) — **but $O(1)$ along $u_c$**: the injective-norm/Frobenius gap. This object *is* the mixed trace $C(v,v,i,j) = T_{4,2}$ of `spike_kprop`, and it is why the trace-projection rule ("keep the $q = K{+}1$ trace") is necessary but its isotropic part alone can miss the $u_c u_c^\top$ piece.

> **Prediction for `e1_cumulant_scaling`:** the *traceless* part of $T_{4,2}$ contains a rank-1 component of squared size $n^0$, sitting above the generic hypothesis $n^{1-q} = n^{-1}$. (Consistent with "R≥3 only halves the e1 error".)

**Closure.** ReLU $\circ$ ($W' + \theta vv^\top$) maps $\mathcal{C} \to \mathcal{C}$: conditionally on $S$ the bulk pre-activations are a weakly coupled field with couplings in $\mathcal{D}(n)$; a coordinatewise nonlinearity keeps it there (linked-cluster counting, §7: random-sign $n^{-1/2}$ couplings; trees give Frobenius $\sqrt n$, coupling-squared loops and the collective mode make traces *saturate* $O(n)$ — the class boundary, which is why the scalings are clean power laws); the spike row plus the coordinatewise kink regenerate the rank-1 knee channel, which the spike-conditional predictors (binned / analytic affine-conditioned) exist to carry.

---

## 1. Conventions

Repo model (matching `e1_cumulant_scaling` / `spike_kprop` builders): hidden layers $M^{(\ell)} = W'^{(\ell)} + \theta v v^\top$, $W'_{ij} \sim N(0, 1/\text{fan-in})$, spike on hidden layers only, readout unspiked, no biases, $X \sim N(0, I_n)$, ReLU $\varphi$, $\theta = \pm 1$ by experiment. $P = I - vv^\top$; $S = v\cdot a$; the measured slices are $T_{r,q} = \kappa_r(S^{r-q}, a_\perp^q)$ with $q$ free transverse legs. Joint cumulants are multilinear and shift-invariant for $r \ge 2$; all "$O(\cdot)$" statements are uniform over the stated order range with constants independent of $n$.

## 2. Step 1 — multilinearity: what a transverse direction sees

For $u \perp v$: $u^\top M a = u^\top W' a = \sum_i w_i a_i$ with $w = W'^\top u$. Cumulants are multilinear, so at fixed $W'$

$$\kappa_r\Big(\sum_i w_i a_i\Big) = \sum_{i_1 \cdots i_r} w_{i_1} \cdots w_{i_r}\, \kappa(a_{i_1}, \dots, a_{i_r}) = T_r[w^{\otimes r}].$$

Because $W'$ is Gaussian and $u$ is a fixed unit vector, $w \sim N(0, I_n/n)$ *exactly* — no CLT needed — and for orthonormal $u^{(1)}, u^{(2)}, \dots$ the probes $w^{(1)}, w^{(2)}, \dots$ are exactly independent. Two consequences worth stating: (i) the spike strength $\theta$ never appears transversally — it enters only through the cumulants of $a$ itself; (ii) everything below is a statement about the *tensors* $T_r$ probed by iid Gaussian vectors, i.e. a property of the input layer's distribution alone.

## 3. Step 2 — the systematic part: Wick mean = fully paired trace

Isserlis on the $r$ probe factors, $\mathbb{E}[w_a w_b] = \delta_{ab}/n$:

$$\mathbb{E}\,T_r[w^{\otimes r}] = n^{-r/2} \sum_{\text{perfect matchings of } r \text{ slots}} (\text{$T_r$ contracted along the matching}) = n^{-r/2}(r-1)!!\; \mathrm{pt}_r$$

for even $r$ (all matchings agree by symmetry of $T_r$), and $0$ for odd $r$. **iid benchmark:** $T_r$ diagonal with entries $\kappa_r$, so $\mathrm{pt}_r = n\kappa_r$ (the trace indices are forced equal), giving $\kappa_r^\perp \approx (r-1)!!\,\kappa_r\, n^{1-r/2}$ — the CLT rate. This is the precise sense of "contracted diagrams are $O(n)$" at one vertex: *the systematic transverse cumulant is $n^{-r/2}$ times a fully contracted diagram; it has the iid size iff that diagram is $O(n)$.*

## 4. Step 3 — the scatter: the variance identity (the keystone)

Second moment: Isserlis over $2r$ slots (two copies of $T_r$, same $w$). A matching with $p$ intra-copy pairs in each copy (counts must match) and $r - 2p$ cross pairs contributes $\langle T_r^{(p)}, T_r^{(p)}\rangle = \|T_r^{(p)}\|_F^2$, independent of which slots were chosen (symmetry). Counting the matchings — $\binom{r}{2p}(2p-1)!!$ per copy, $(r-2p)!$ cross bijections — and subtracting the fully-intra terms ($=$ mean$^2$):

$$\mathrm{Var}\,T_r[w^{\otimes r}] = n^{-r}\sum_{p=0}^{\lceil r/2\rceil-1} \left[\binom{r}{2p}(2p-1)!!\right]^2 (r-2p)!\; \|T_r^{(p)}\|_F^2.$$

Worked low orders (all MC-verified in the companion script): $\mathrm{Var}(w^\top A w) = 2\|A\|_F^2/n^2$; $\mathrm{Var}\,T_3[w^{\otimes 3}] = n^{-3}(6\|T_3\|_F^2 + 9\|T_3^{(1)}\|^2)$ with $T^{(1)}_{3,i} = \sum_j T_{ijj}$; $\mathrm{Var}\,T_4[w^{\otimes 4}] = n^{-4}(24\|T_4\|_F^2 + 72\|T_4^{(1)}\|_F^2)$.

**Why this identity settles the characterization.** All coefficients are strictly positive, so the variance is small *iff every* $\|T_r^{(p)}\|_F$ is small — no conspiracy among functionals can hide a large one. Combined with §3: the boxed iff of R2. Sufficiency and necessity are both at the level of the first two $W$-moments, which is the operationally right notion ("typical seed, typical output coordinate"): if some functional exceeds its $O(n)$ budget by a factor $\lambda$, either the systematic part or the RMS over seeds/coordinates exceeds the iid size by $\lambda$.

## 5. Step 4 — all orders and moments: the diagram class

For $2m$-th $W$-moments the same computation produces contractions of $2m$ copies of $T_r$; matchings decompose into connected clusters, and moments factorize over clusters up to combinatorial constants. Define therefore (R3) the class $\mathcal{D}(n)$: *every closed connected contraction diagram of connected cumulant tensors (including $\mu$ as an order-1 vertex) is $O(n)$.* Then all $W$-moments of all transverse cumulants obey the iid-Gaussian-chaos sizes. Disconnected diagrams factorize, so they are automatically $O(n^{\#\text{components}})$; the class is genuinely a constraint per connected component.

**Saturation lemma (why "O(n)" is the right boundary):** for iid coordinates every tensor is diagonal; in a connected diagram the contractions force a single shared summation index, so every connected diagram is exactly $n \prod_v \kappa_{r_v}$. iid is *extremal*: it saturates every diagram budget simultaneously. Post-ReLU layers of the spiked net sit at this boundary too, but through a different mechanism (couplings, §7), which is why extensivity ($\sim n^1$, not $\ll n$) is the observed clean law in `normsq_cumulant_scaling`.

**Odd orders get a free $\sqrt n$.** For odd $r$ the mean vanishes and the size is pure scatter $n^{(1-r)/2}$: contrast a *deterministic all-positive* probe like $w = \mathbf{1}/\sqrt n$ on an iid vector, where $\kappa_3 = \sum_i n^{-3/2}\kappa_3 = n^{-1/2}\kappa_3$ — no sign cancellation. Generic (Gaussian-row) directions are sign-incoherent; structured deterministic directions are not. This is precisely why the *spike component* (the one structured direction the model insists on) must be tracked exactly, and it is the transverse/spike dichotomy in one line.

## 6. Step 5 — the annealed collapse: locating the $\|a\|^2$ criterion

Average over $W'$ *jointly* with $a$: conditional on $a$, $u\cdot h \sim N(0, \|a\|^2/n)$ exactly. A Gaussian scale mixture has cumulant generating function $K(t) = K_V(t^2/2)$, $V = \|a\|^2/n$, hence

$$\kappa_{2r}^{\mathrm{ann}}(u \cdot h) = \frac{(2r)!}{2^r r!}\,\kappa_r(V) = (2r-1)!!\,\frac{\kappa_r(\|a\|^2)}{n^r}, \qquad \kappa_{\mathrm{odd}}^{\mathrm{ann}} = 0.$$

So "$\kappa_r(\|a\|^2)$ extensive $\Rightarrow$ annealed transverse cumulants iid-sized" is exact and *is* your earlier criterion. Expanding $\kappa_r(\|a\|^2) = \sum_{i_1\cdots i_r} \kappa(a_{i_1}^2, \dots, a_{i_r}^2)$ by Leonov–Shiryaev (sum over partitions connecting the $r$ squares, unit coefficients) rewrites it as a positively-weighted sum over closed connected diagrams with $r$ edges — e.g. $\kappa_2(\|a\|^2) = \mathrm{pt}_4 + 2\|\Sigma\|_F^2 + 4\,\mu^\top T_3^{(1)\top}\!\cdots$-type terms. One-way implication follows; the three failure modes of the converse are R4(1–3). The operational fix is to control (or measure — §9) the diagrams individually; that is what the quenched layer actually consumes.

## 7. Step 6/7 — the spike, the rank-1 anomaly, and closure

**Conditioning.** Condition on $S$. Brillinger's total-cumulance formula: each unconditional $\kappa^{(r)}$ is the sum over partitions of cumulants (over $S$) of conditional cumulants. If conditionally the bulk is in $\mathcal{D}(n)$ and the $S$-dependence of the conditional cumulants is carried by $O(1)$-many smooth channels — empirically one: $m(S) \approx \text{affine} + g(S) u_c$ (`affine_conditional_layer1` knee; conditional covariance nearly $S$-flat per `empirical_structure`) — then the unconditional tensors are

$$\kappa^{(r)}_{\mathrm{uncond}} = \underbrace{\mathbb{E}_S\,\kappa^{(r)}_{\mathrm{cond}}}_{\in\, \mathcal{D}(n)} \;+\; \underbrace{\text{terms like } \kappa_j(g)\, u_c^{\otimes j} \otimes (\text{lower conditional tensors})}_{\text{rank-}O(1),\ \text{Frobenius and traces } O(1)}.$$

The anomaly is invisible to every aggregate in R2–R4 (an $O(1)$ perturbation of $O(n)$ budgets) and dies through fresh generic legs ($\langle u_c, w\rangle \sim n^{-1/2}$ each), so *the bulk stays clean layer after layer* — this is the actual reason "the coordinates are somewhat independent after a linear layer, even with the shift". But along $u_c$ it is $O(1)$: unconditional bulk Gaussianity fails in exactly one direction per channel. Direction-resolved objects see it (the $T_{4,2}$ mixed trace; the width-flat layer-2 knee residual); aggregates and norms don't. Conditioning on $S$ (binned / HMM / affine-conditioned predictors) removes it by construction; the trace-projection theorem says which of its contractions must be retained at budget $K$, and the $u_cu_c^\top$-in-the-traceless-bin prediction (R5) sharpens that rule.

**ReLU closure (counted, not proved).** Conditionally on $S$, bulk pre-activations $h_\perp$ are a Gaussian-dominated field: coordinate pairs couple at $\rho_{ij} = O(n^{-1/2})$ with random signs (row overlaps through $\Sigma_a$), plus non-Gaussian vertices already in $\mathcal{D}(n)$ by induction. A coordinatewise $\varphi$ maps connected cumulants of distinct coordinates to sums over connected interaction diagrams: vertices carry $O(1)$ Hermite weights of $\varphi$ at the local $(\mu_i, \sigma_i)$, links carry the small couplings. Power count: spanning trees over $r$ coordinates give entries $\sim \rho^{r-1}$, so $\|\kappa^{(r)}\|_F^2 \lesssim n^r \cdot n^{-(r-1)} = n$ (two-vertex budget); paired traces are dominated by coupling-squared chains and the collective mode, $\sum_{ij} \rho_{ij}^2 \sim n$ and $\mathbf{1}^\top R \mathbf{1} \sim n$ (one-vertex budget saturated). Empirically (§10) the traces sit cleanly on the boundary at both depths, while the odd Frobenius mass is boundary-level at layer 1 and *sub*-extensive (slope $\approx 0.65$) at layer 2 — sign cancellation in the tree sums is partially helping; the class only requires $\le$. The spike row plus the ReLU kink regenerate the knee channel ($g$ is the conditional-mean response of $\varphi$ to the $S$-coherent shift — the Tweedie knee), i.e. the $\oplus$-part of $\mathcal{C}$ is reproduced with rank staying $O(1)$. Status: the linear half (§§2–5) is exact; this half is a power-counting argument whose conclusions are exactly what `normsq_cumulant_scaling` (traces, $r{=}2,3,4$) and `e1_cumulant_scaling` (directional slices) are built to test, plus the model-level checks in the companion script. Known boundary case: the *flat* spike's collective mode makes the *unconditional* bulk leave $\mathcal{D}(n)$ (`ones(+)` self-similar $\kappa_r \sim \kappa_2^{r/2}$) — consistent with $\mathcal{C}$ being conditional; for $v = e_1$ the collective coordinate *is* $S$ and conditioning restores the class.

## 8. Where the Edgeworth attempt fits

Reconstructing densities/scores from cumulants (the Gram–Charlier / Edgeworth formulas) is downstream of this note: once transverse cumulants have the sizes in R3's table, standardized cumulants obey $\lambda_r = O(n^{1-r/2})$ ($O(n^{(1-r)/2})$ odd) and Edgeworth is a genuine asymptotic expansion in $n^{-1/2}$ — truncation error smaller than the last kept term. Outside the class ($O(1)$ cumulants, e.g. the sub-shift death regime) the expansion has no small parameter and diverges — exactly the observed swkprop Edgeworth blow-up. So the characterization is the *license* for any cumulant-based density surgery, not a competitor to it.

## 9. Measurement dictionary

| functional | estimator | where |
|---|---|---|
| $\mathrm{pt}_4 = \sum_{ij}\kappa_4(a_i,a_i,a_j,a_j)$ | probe pairs $g,g' \sim N(0,I)$: $\mathbb{E}_{g,g'}\,\hat\kappa_4(\langle g,c\rangle,\langle g,c\rangle,\langle g',c\rangle,\langle g',c\rangle) = \mathrm{pt}_4$ | companion script (new) |
| $\|T_3\|_F^2$, $\|T_3^{(1)}\|^2$ | probe triples/singles + split-half cross product $\langle \hat T^A, \hat T^B\rangle$ (kills the $+$noise bias) | same trick as `e1_cumulant_scaling` `crossv/crossm` |
| $\kappa_r(\|a\|^2)$ (annealed aggregate) | `analysis/Tools/cumulants_sv.py` streamer | `normsq_cumulant_scaling` |
| directional slices $T_{r,q}$, trace vs traceless | $P$-projected slice tensors, split-half cross | `e1_cumulant_scaling` |
| downstream check $\kappa_4(h_b) \approx 3\,\mathrm{pt}_4/n^2$, $\kappa_3(h_b)$ RMS $= n^{-3/2}\sqrt{6\|T_3\|_F^2 + 9\|T_3^{(1)}\|^2}$ | direct per-coordinate cumulants at the next layer vs input-side functionals | companion script (closes the loop) |

Diagnostics this suggests: (i) the ratio scatter/systematic per order — distance from the class boundary; (ii) the traceless-$T_{4,2}$ rank-1 test (R5 prediction); (iii) if the width-law exponent drifts from $-2$ toward $-1$ (`widthlaw_significance`), suspect a functional leaving its budget — the family above says *which* to check.

## 10. Verification status — 21/21 checks passed (2026-07-17)

`verify_transverse_pass.py`; numbers in `experiments/transverse_pass_characterization/stats_cache.json`.

| claim | test | status |
|---|---|---|
| mean identity (§3), $r=2,3,4$ | dense-tensor MC vs formula | **PASS** (all $z<1$) |
| variance identity (§4) coefficients $(2),(6,9),(24,72)$ | dense-tensor MC vs formula | **PASS** (rel err 0.1–0.5%) |
| annealed identity (§6), $r=2$ | joint MC vs $3\kappa_2(\|a\|^2)/n^2$ | **PASS** (rel 2.1%, $z=1.6$) |
| odd-order quenched/annealed gap (R4.2) | per-seed $\kappa_3^\perp$ scatter vs exact $T_3$-functional formula; pooled $\to 0$ | **PASS** (sd ratio 0.85 in 24-seed band; per-seed span $[-0.40,+0.56]$ vs annealed 0) |
| $t_2, \mathrm{pt}_4, \|T_3\|_F^2, \|T_3^{(1)}\|^2 = O(n)$ on the e1 net | probe estimators, widths 64–256, depth 3, $\theta=-1$, seeds 1–3 | **PASS** — slopes L1: 1.00 / 1.10 / 1.08 / 1.16; L2: 0.98 / 1.02 / 0.65 / 1.21 (traces saturate the boundary; deep odd Frobenius runs *sub*-extensive) |
| downstream $\kappa_4(h_b) = 3\,\mathrm{pt}_4/n^2$ | layer $\ell \to \ell{+}1$, 12 coords/seed | **PASS** — geomean ratio 1.04 (e.g. $n{=}256$, L2: 0.00116 vs 0.00118) |
| downstream $\mathrm{RMS}\,\kappa_3(h_b) = n^{-3/2}\sqrt{6\|T_3\|_F^2 + 9\|T_3^{(1)}\|^2}$ | same | **PASS** — geomean ratio 1.08 |
| $\kappa_2(h_b) = t_2/n$ | same | **PASS** — geomean 1.02 |

## 11. Open items

1. Necessity at $W$-moments $m \ge 3$ is stated per-diagram-family but proved only through the variance level; higher moments need a positivity argument per cluster size (or accept RMS-level necessity, which is the operative one).
2. An explicit distribution where $\kappa_r(\|a\|^2) = O(n)$ for all $r$ but some individual diagram is $\gg n$ (cancellation in the L–S aggregate) — the odd-order gap (verified) already shows the annealed criterion is strictly weaker; a fully even-order cancellation example would close the taxonomy.
3. Rank growth of the anomaly channel with depth: is it exactly rank-1 per layer or does the knee spawn $O(\text{depth})$ channels ($\mathcal{C}$ tolerates $O(1)$)? `empirical_structure`'s "diagonal not rank-1 later" hints at a second (variance) channel — measurable via the $T_{4,2}$ eigen-spectrum.
4. Uniformity in $S$ of the conditional class in the far tails (Tweedie knee flattens — likely fine, unchecked).

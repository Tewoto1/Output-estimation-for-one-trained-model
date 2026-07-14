"""Generates rebinning_mechanism_colab.ipynb -- SELF-CONTAINED (all helper code inline).

Mechanistic analysis of WHERE binned-kprop's linear (rebinning) step loses accuracy, for the
coordinate-spike case M = W + e1 e1^T, plus the WITHIN-BIN-VARIANCE FIX and its measured gain.

The linear step (core.linear_step_k2) models the new spike Y = gamma*A + r.B per old bin as a
Gaussian N(gamma a_alpha + r.mu_alpha, r Sigma_alpha r^T) integrated over the new edges. Two errors:
  collapse    -- A set to a point => Var(Y|a) modelled as Var(r.B|a), DROPS gamma^2 Var(A|a) (gamma~1);
                 a bin-resolution error ~1/num_bins, width-independent.
  gaussianity -- Y|a assumed Gaussian; truly ~Gaussian by CLT up to skewness ~ n^-1/2; vanishes with
                 width, grows with depth.
The mean is modelled exactly, so algo vs truth differ only in variance (collapse) and shape (gaussianity).
This notebook measures both from the scalars Y and P=r.B (no d x d -> cheap on T4/L4, fp32), and tests
the FIX: put gamma^2 Var(A|a) back into the new-spike variance.

Run:  python "experiments/rebinning_mechanism/build_rebinning_mechanism_notebook.py"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _nb import NotebookBuilder, BOOTSTRAP_CELL

nb = NotebookBuilder()
md, code = nb.md, nb.code

md(r"""# Rebinning-step error: **collapse vs non-Gaussianity**, and the within-bin-variance fix

**Case:** `M = W + e1 e1^T` (e1 e1^T shift of a random `N(0,1/n)` matrix, no training), `X ~ N(0,I)`, ReLU.

**The step under test** (`core.linear_step_k2`). Going into a linear layer the state is binned along the
spike coord. For each old bin `α` (representative spike `a_α`, bulk mean `μ_α`, bulk cov `Σ_α`) the new
spike

$$Y = z'_0 = \gamma A + r\cdot B,\qquad \gamma=M_{00}\approx 1,\quad r=M_{0,1:}\sim 1/\sqrt n,$$

is modelled as **Gaussian** `N(γa_α + r·μ_α, r Σ_α rᵀ)` and integrated over the new bin edges to get the
transition probabilities. Trace the logic layer by layer:

- **input:** `X ~ N(0,I)` → bins have identical bulk cov (nothing lost);
- **after a linear map:** jointly Gaussian → conditional bulk cov is bin-independent (exact);
- **after ReLU:** each bin's bulk is `ReLU(Gaussian)` — non-Gaussian — but its mean & cov are tracked
  *exactly* (exact Gaussian integral); only higher cumulants are dropped;
- **the next rebinning is where it breaks**, via exactly two approximations:

| term | what | scaling | vanishes with |
|---|---|---|---|
| **collapse** | `A`→point `a_α`, so `Var(Y\|α)` uses `Var(r·B\|α)` only, **dropping `γ²Var(A\|α)`** (γ≈1!) | ~1/num_bins (tail-limited) | **bins**, not width |
| **gaussianity** | `Y\|α` assumed Gaussian; truly a `1/√n`-sum of non-Gaussian bulk | `skew_P ~ n^(−1/2)` | **width**, not bins; grows with depth |

The **mean is exact** (`γa_α+r·μ_α = E[Y|α]`), so algo vs truth differ only in **variance** (collapse) and
**shape** (gaussianity). We compare four models of `Y|α` over the new bins:

    Q_algo = N(E[Y|α], Var(P|α))                      (current: collapse variance)
    Q_fix  = N(E[Y|α], Var(P|α) + γ² Var(A|α))        (THE FIX: put the within-bin spike variance back)
    Q_best = N(E[Y|α], Var(Y|α))                       (exact variance; only shape left)
    Q_emp  = MC histogram
        TV(Q_algo, Q_emp)  total error today
        TV(Q_fix,  Q_emp)  error after the fix        (gain = TV_algo − TV_fix)
        TV(Q_best, Q_emp)  non-Gaussianity floor      (irreducible without K>2)

Everything is scalar per sample (`Y`, `P=r·B`, `A`), so no `d×d` — it streams cheaply on a T4/L4 in fp32.""")

code(r"""!pip install -q scipy""")
code(BOOTSTRAP_CELL)

# =============================================================================
md(r"""## Config""")
code(r"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtr                       # standard-normal CDF (Phi)
import experiments as E
from Mecha_preds.binned_kprop import build_spiked_net    # kept module (randn/sqrt(n) + e1e1^T)

QUICK = E.QUICK
def _envlist(name, default):
    v = os.environ.get(name, "")
    return [int(x) for x in v.split(",") if x.strip()] or default

WIDTHS   = _envlist("REB_WIDTHS", [48, 96, 192] if QUICK else [64, 128, 256, 512, 1024])
SEEDS    = _envlist("REB_SEEDS", [1, 2] if QUICK else [1, 2, 3, 4])
DEPTH    = int(os.environ.get("REB_DEPTH", 3 if QUICK else 4))    # deeper -> see accumulation
NUM_BINS = int(os.environ.get("REB_BINS", 15 if QUICK else 21))
THETA, OUT_DIM, MC_SEED = 1.0, 8, 7
N_SAMPLES = int(os.environ.get("REB_SAMPLES", 500_000 if QUICK else 4_000_000))
N_PILOT   = min(200_000, N_SAMPLES // 3)
BATCH     = int(os.environ.get("REB_BATCH", 100_000 if QUICK else 250_000))
BINSWEEP  = _envlist("REB_BINSWEEP", [9, 17, 33] if QUICK else [9, 17, 33, 65])
BINSWEEP_N = WIDTHS[-1]
CKPT_DIR  = "checkpoints/rebinning_mechanism"; os.makedirs(CKPT_DIR, exist_ok=True)

try:
    import torch; BACKEND = "torch" if torch.cuda.is_available() else "numpy"
except Exception:
    BACKEND = "numpy"
print(f"QUICK={QUICK} widths={WIDTHS} depth={DEPTH} bins={NUM_BINS} seeds={SEEDS} "
      f"MC={N_SAMPLES:,} batch={BATCH:,} backend={BACKEND}")
""")

# =============================================================================
md(r"""## Inline scan (self-contained) — Y, P=r·B, A per old bin; four models of `Y|α`

Scalar streaming: at each hidden layer bin the pre-activation spike into equal-mass bins; for each
transition `ℓ→ℓ+1` accumulate per old bin the moments of `Y` (new spike), `P=r·B` (bulk projection) and
`A` (old post-ReLU spike), plus the joint (old,new)-bin histogram. `fp32` compute, `fp64` accumulators.""")
code(r"""
def _gauss_bin_probs(mean, var, edges):
    s = float(np.sqrt(max(var, 1e-12)))
    return np.clip(np.diff(ndtr((edges - mean) / s)), 0.0, None)

def _cmom(c, s1, s2, s3, s4):
    m = s1/c; e2=s2/c; e3=s3/c; e4=s4/c
    return m, e2-m*m, e3-3*m*e2+2*m**3, e4-4*m*e3+6*m*m*e2-3*m**4

def _edges(x, nb):
    e = np.maximum.accumulate(np.quantile(x, np.linspace(0,1,nb+1))); e[0], e[-1] = -np.inf, np.inf
    for i in range(1,len(e)-1):
        if e[i] <= e[i-1]: e[i] = np.nextafter(e[i-1], np.inf)
    return e

_KEYS = ["cnt","Y1","Y2","Y3","Y4","P1","P2","P3","P4","A1","A2"]
def _blank(nb):
    d = {k: np.zeros(nb) for k in _KEYS}; d["C"] = np.zeros((nb,nb)); return d

def _scan_numpy(Wh, n, nb, edges, N, batch, seed, sc=0):
    L=len(Wh); bulk=np.array([i for i in range(n) if i!=sc]); acc=[_blank(nb) for _ in range(L-1)]
    rng=np.random.default_rng(seed); got=0
    while got<N:
        b=min(batch,N-got); h=rng.standard_normal((b,n)); pb=ps=phb=None
        for li in range(L):
            z=h@Wh[li].T; sp=z[:,sc]; cb=np.clip(np.searchsorted(edges[li],sp,side="right")-1,0,nb-1)
            if li>=1:
                a=acc[li-1]; r=Wh[li][sc,bulk]; P=phb@r; Y=sp; A=np.maximum(ps,0.0)
                np.add.at(a["cnt"],pb,1.0)
                for k in range(1,5): np.add.at(a["Y%d"%k],pb,Y**k); np.add.at(a["P%d"%k],pb,P**k)
                for k in range(1,3): np.add.at(a["A%d"%k],pb,A**k)
                np.add.at(a["C"],(pb,cb),1.0)
            pb=cb; ps=sp; phb=np.maximum(z[:,bulk],0.0); h=np.maximum(z,0.0)
        got+=b
    return acc

def _scan_torch(Wh_np, n, nb, edges_np, N, batch, seed, sc=0):
    import torch
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); dt=torch.float32
    L=len(Wh_np); Wh=[torch.as_tensor(W,dtype=dt,device=dev) for W in Wh_np]
    bulk=torch.as_tensor([i for i in range(n) if i!=sc],device=dev)
    edges=[torch.as_tensor(e,dtype=dt,device=dev).contiguous() for e in edges_np]
    acc=[{k:torch.zeros(nb,dtype=torch.float64,device=dev) for k in _KEYS} for _ in range(L-1)]
    for a in acc: a["C"]=torch.zeros(nb*nb,dtype=torch.float64,device=dev)
    g=torch.Generator(device=dev).manual_seed(seed); got=0
    while got<N:
        b=min(batch,N-got); h=torch.randn(b,n,generator=g,dtype=dt,device=dev); pb=ps=phb=None
        for li in range(L):
            z=h@Wh[li].T; sp=z[:,sc]
            cb=torch.clamp(torch.searchsorted(edges[li],sp.contiguous(),right=True)-1,0,nb-1)
            if li>=1:
                a=acc[li-1]; P=(phb@Wh[li][sc].index_select(0,bulk)).double(); Y=sp.double()
                A=torch.relu(ps).double(); ob=pb
                a["cnt"].index_add_(0,ob,torch.ones_like(Y))
                a["Y1"].index_add_(0,ob,Y); a["Y2"].index_add_(0,ob,Y*Y); a["Y3"].index_add_(0,ob,Y**3); a["Y4"].index_add_(0,ob,Y**4)
                a["P1"].index_add_(0,ob,P); a["P2"].index_add_(0,ob,P*P); a["P3"].index_add_(0,ob,P**3); a["P4"].index_add_(0,ob,P**4)
                a["A1"].index_add_(0,ob,A); a["A2"].index_add_(0,ob,A*A)
                a["C"].index_add_(0,ob*nb+cb,torch.ones_like(Y))
            pb=cb; ps=sp; phb=torch.relu(z.index_select(1,bulk)); h=torch.relu(z)
        got+=b
    out=[]
    for a in acc:
        d={k:a[k].cpu().numpy() for k in _KEYS}; d["C"]=a["C"].cpu().numpy().reshape(nb,nb); out.append(d)
    return out

def _finalize(a, new_edges, gamma):
    cnt=a["cnt"]; tot=cnt.sum(); p=cnt/tot if tot>0 else cnt
    tv_algo=tv_fix=tv_best=skP=coll=wsum=0.0
    for al in range(cnt.shape[0]):
        c=cnt[al]
        if c<=1: continue
        mY,mu2Y,_,_=_cmom(c,a["Y1"][al],a["Y2"][al],a["Y3"][al],a["Y4"][al])
        _,mu2P,mu3P,_=_cmom(c,a["P1"][al],a["P2"][al],a["P3"][al],a["P4"][al])
        mA=a["A1"][al]/c; varA=max(a["A2"][al]/c-mA*mA,0.0)
        if mu2Y<=1e-12: continue
        qe=a["C"][al]/c
        qa=_gauss_bin_probs(mY,mu2P,new_edges)                        # collapse
        qf=_gauss_bin_probs(mY,mu2P+gamma*gamma*varA,new_edges)       # FIX
        qb=_gauss_bin_probs(mY,mu2Y,new_edges)                        # exact variance
        w=p[al]; wsum+=w
        tv_algo+=w*0.5*np.abs(qa-qe).sum(); tv_fix+=w*0.5*np.abs(qf-qe).sum(); tv_best+=w*0.5*np.abs(qb-qe).sum()
        if mu2P>1e-12: skP+=w*abs(mu3P/mu2P**1.5)
        coll+=w*(1-mu2P/mu2Y)
    z=wsum+1e-30
    return dict(tv_algo=tv_algo/z, tv_fix=tv_fix/z, tv_best=tv_best/z, skew_P=skP/z, collapse_frac=coll/z)

def rebinning_scan(Ws, n, num_bins, n_samples, n_pilot, batch, seed, backend="numpy"):
    Wh=[np.asarray(W,dtype=np.float64) for (W,_b) in Ws][:-1]; L=len(Wh)
    if L<2: return []
    # pilot -> equal-mass edges per layer on the pre-activation spike
    rng=np.random.default_rng(seed+999); cols=[[] for _ in range(L)]; got=0
    while got<n_pilot:
        b=min(batch,n_pilot-got); h=rng.standard_normal((b,n))
        for li in range(L): z=h@Wh[li].T; cols[li].append(z[:,0].copy()); h=np.maximum(z,0.0)
        got+=b
    edges=[_edges(np.concatenate(cols[li]),num_bins) for li in range(L)]
    gammas=[float(Wh[t+1][0,0]) for t in range(L-1)]
    if backend=="torch":
        try: acc=_scan_torch(Wh,n,num_bins,edges,n_samples,batch,seed)
        except Exception as e: print("  (torch->numpy):",e); acc=_scan_numpy(Wh,n,num_bins,edges,n_samples,batch,seed)
    else:
        acc=_scan_numpy(Wh,n,num_bins,edges,n_samples,batch,seed)
    return [_finalize(acc[t], edges[t+1], gammas[t]) for t in range(L-1)]
print("inline scan ready")
""")

# =============================================================================
md(r"""## Run — width×seed sweep (fixed bins) + num_bins sweep (fixed width)""")
code(r"""
def run_cached(n, depth, nb, N, sd):
    k=os.path.join(CKPT_DIR, f"reb_d{depth}_w{n}_nb{nb}_s{sd}_S{N}.npz")
    if os.path.exists(k):
        z=np.load(k, allow_pickle=True); return list(z["tr"])
    Ws=build_spiked_net(n, depth, seed=sd, theta=THETA, out_dim=OUT_DIM)
    tr=rebinning_scan(Ws, n, nb, N, N_PILOT, BATCH, MC_SEED+sd, backend=BACKEND)
    np.savez(k, tr=np.array(tr, dtype=object)); return tr

reb={}   # (n, seed) -> per-transition dicts (fixed NUM_BINS)
for n in WIDTHS:
    for sd in SEEDS:
        t0=time.time(); reb[(n,sd)]=run_cached(n, DEPTH, NUM_BINS, N_SAMPLES, sd)
        print(f"  d{DEPTH} n{n} s{sd}: {time.time()-t0:.1f}s")
rebbins={nb: run_cached(BINSWEEP_N, DEPTH, nb, N_SAMPLES, SEEDS[0]) for nb in BINSWEEP}
print("done. transitions per net:", DEPTH-1)
""")

# =============================================================================
md(r"""## The fix's gain + the scalings (exact numbers)""")
code(r"""
nT = DEPTH - 1
print(f"=== TV over the new bins, n={WIDTHS[-1]}, depth {DEPTH}, {NUM_BINS} bins (seed {SEEDS[0]}) ===")
print(f"{'ell->+1':>8} | {'TV_algo':>8} {'TV_fix':>8} {'TV_best':>8} | {'gain(algo-fix)':>14} | {'skew_P':>7} {'collapse%':>9}")
for t,r in enumerate(reb[(WIDTHS[-1],SEEDS[0])]):
    print(f"{t:>3}->{t+1:<3} | {r['tv_algo']:8.4f} {r['tv_fix']:8.4f} {r['tv_best']:8.4f} | "
          f"{r['tv_algo']-r['tv_fix']:14.4f} | {r['skew_P']:7.3f} {100*r['collapse_frac']:8.1f}%")
print("  TV_fix should sit near TV_best (=non-Gaussianity floor): the fix removes the collapse term.")

lo=np.log(np.array(WIDTHS,float))
print("\n=== NON-GAUSSIANITY skew_P vs width (seed-avg; the irreducible-without-width part) ===")
for t in range(nT):
    ym=np.array([np.mean([reb[(n,sd)][t]["skew_P"] for sd in SEEDS]) for n in WIDTHS])
    a=np.polyfit(lo,np.log(np.clip(ym,1e-6,None)),1)[0]
    print(f"  {t}->{t+1}: "+" ".join(f"{v:.3f}" for v in ym)+f"   alpha={a:+.2f}  (predict ~ -0.5)")

print(f"\n=== num_bins tradeoff at n={BINSWEEP_N} (last transition): collapse-driven TV_algo falls, fix flat ===")
print(f"  {'bins':>5} | {'TV_algo':>8} {'TV_fix':>8} {'TV_best':>8}")
for nb in BINSWEEP:
    r=rebbins[nb][nT-1]; print(f"  {nb:>5} | {r['tv_algo']:8.4f} {r['tv_fix']:8.4f} {r['tv_best']:8.4f}")
""")

# =============================================================================
md(r"""## Plots""")
code(r"""
Wn=np.array(WIDTHS,float); fig,ax=plt.subplots(1,3,figsize=(16,4.4))
# (1) skew_P vs width, per transition, with n^-1/2 ref
for t in range(nT):
    sk=np.array([np.mean([reb[(n,sd)][t]["skew_P"] for sd in SEEDS]) for n in WIDTHS])
    ax[0].loglog(Wn, np.clip(sk,1e-6,None), "o-", label=f"{t}->{t+1}")
ax[0].loglog(Wn, sk[0]*(Wn/Wn[0])**-0.5, "k:", alpha=.6, label="n^-1/2")
ax[0].set_title("non-Gaussianity skew_P vs width", fontsize=10); ax[0].set_xlabel("width n"); ax[0].legend(fontsize=7); ax[0].grid(True,which="both",alpha=.25)
# (2) the fix: TV_algo vs TV_fix vs TV_best per transition (at largest width)
r_last=reb[(WIDTHS[-1],SEEDS[0])]; x=np.arange(nT); wdi=0.25
ax[1].bar(x-wdi,[r_last[t]["tv_algo"] for t in range(nT)],wdi,label="TV_algo (now)")
ax[1].bar(x,     [r_last[t]["tv_fix"]  for t in range(nT)],wdi,label="TV_fix")
ax[1].bar(x+wdi,[r_last[t]["tv_best"] for t in range(nT)],wdi,label="TV_best (floor)")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{t}->{t+1}" for t in range(nT)])
ax[1].set_title(f"the fix's gain (n={WIDTHS[-1]})", fontsize=10); ax[1].set_ylabel("TV over new bins"); ax[1].legend(fontsize=7); ax[1].grid(axis="y",alpha=.25)
# (3) num_bins tradeoff
bb=np.array(BINSWEEP,float)
ax[2].loglog(bb,[rebbins[b][nT-1]["tv_algo"] for b in BINSWEEP],"o-",label="TV_algo")
ax[2].loglog(bb,[rebbins[b][nT-1]["tv_fix"]  for b in BINSWEEP],"s-",label="TV_fix")
ax[2].loglog(bb,[rebbins[b][nT-1]["tv_best"] for b in BINSWEEP],"^-",label="TV_best")
ax[2].loglog(bb, rebbins[BINSWEEP[0]][nT-1]["tv_algo"]*(bb/bb[0])**-1.0, "k:", alpha=.5, label="1/bins")
ax[2].set_title(f"num_bins tradeoff (n={BINSWEEP_N})", fontsize=10); ax[2].set_xlabel("num_bins"); ax[2].legend(fontsize=7); ax[2].grid(True,which="both",alpha=.25)
fig.suptitle("Rebinning step: fix removes the collapse term; non-Gaussianity floor falls with width", y=1.02)
fig.tight_layout(); plt.show()
""")

# =============================================================================
md(r"""## Verdict + how to apply the fix in `core.linear_step_k2`

**Verdict (auto-printed below).** If `TV_fix ≈ TV_best < TV_algo`, the within-bin-variance fix removes the
collapse error and what remains is the non-Gaussianity floor, which only width (or `K>2`) lowers.

**The one change in `core.linear_step_k2`.** Today `sY2[alpha] = r @ Sig_a @ r` is `Var(r·B|α)` only.
The fix adds the within-bin spike variance (which the *previous* rebinning already computes as the
truncated-normal variance `yvar[beta, alpha]`): carry a per-bin spike variance `svar_α` in the state
(one extra `(num_bins,)` field on `BinnedK2State`, set from `yvar` at the end of `linear_step_k2`), then

```
sY2[alpha] = float(r @ Sig_r) + (gamma ** 2) * svar[alpha]   # + within-bin spike variance
```

That is the entire behavioural change; it costs one length-`num_bins` array and no extra `O(d^…)` work.""")
code(r"""
r_ref=reb[(WIDTHS[-1],SEEDS[0])]
a_sk=np.polyfit(lo,np.log(np.clip(np.array([np.mean([reb[(n,sd)][nT-1]["skew_P"] for sd in SEEDS]) for n in WIDTHS]),1e-6,None)),1)[0]
gain=np.mean([r_ref[t]["tv_algo"]-r_ref[t]["tv_fix"] for t in range(nT)])
resid=np.mean([r_ref[t]["tv_fix"]-r_ref[t]["tv_best"] for t in range(nT)])
print("VERDICT (rebinning step)")
print(f"  FIX: mean TV gain (algo->fix) = {gain:.4f}; residual above the non-Gaussianity floor (fix-best)"
      f" = {resid:.4f} (~0 => fix recovers the exact-variance model up to the small spike-bulk cross-cov).")
print(f"  Non-Gaussianity floor: skew_P ~ n^{a_sk:+.2f} -> vanishes with WIDTH; grows with depth.")
print(f"  Collapse today: ~{100*np.mean([r_ref[t]['collapse_frac'] for t in range(nT)]):.0f}% of Var(Y) dropped "
      f"at {NUM_BINS} bins -> that is exactly what the fix restores.")
print("  So: apply the fix to kill the collapse term cheaply; use WIDTH (or K>2) to lower the remaining floor.")
""")

nb.save(os.path.join(os.path.dirname(__file__), "rebinning_mechanism_colab.ipynb"))

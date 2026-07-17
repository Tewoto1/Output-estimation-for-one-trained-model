"""Depth propagation of the knee: MC on depth-3 spiked nets (M_l = W_l + e1e1^T).

At each hidden pre-activation layer l in {2,3}, bin the spike a_l = Z^l_0 into
equal-mass bins (edges frozen from a pilot batch) and accumulate SPLIT-HALF
(alternating batches) per-bin sums of
  * the bulk pre-activation  Z^l_{1:}   (PRE curves),
  * the bulk post-activation ReLU(Z^l_{1:})  (POST curves; positive-node fits),
plus the exact channel decomposition into the NEXT spike,
  A^{l+1} = c_l * ReLU(a_l) + xi_l,   c_l = M_{l+1}[0,0],  xi_l = M_{l+1}[0,1:] @ ReLU(Z^l_{1:}),
via per-bin sums of xi and xi^2 (-> mu_l, om2_l, and constancy of Var(xi|a)), and fine
histograms of the spike marginals at l = 1,2,3 (for the 1-d marginal recursion / score test).

Resumable: saves a .partial state and stops after BUDGET_S seconds (env, default 27);
rerun until it reports done. Usage:
    python experiments/affine_conditional_layer1/knee_depth_mc.py [n] [seed] [N_MC]
"""
import os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
CKPT = os.path.join(REPO, "checkpoints", "affine_conditional_layer1")
from Mecha_preds.binned_kprop.empirical_structure import build_spiked_net

n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_MC = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
DEPTH, NBINS, HB = 3, 24, 241
BATCH = min(60_000, max(20_000, 2 ** 24 // n))
BUDGET = float(os.environ.get("BUDGET_S", 27))

out_path = os.path.join(CKPT, f"knee3_mc_n{n}_seed{seed}_N{N_MC}_B{NBINS}.npz")
state_path = out_path + ".partial.npz"
if os.path.exists(out_path):
    print("already complete:", out_path); sys.exit(0)

Ws = build_spiked_net(n, DEPTH, seed=seed)
Ms = [W.astype(np.float32) for W, _ in Ws[:DEPTH]]
cs = np.array([Ws[l + 1][0][0, 0] for l in range(DEPTH - 1)])   # c_2->3 channel consts... c[l] pairs layer l+1
ws = [Ws[l + 1][0][0, 1:].astype(np.float32) for l in range(DEPTH - 1)]

if os.path.exists(state_path):
    st = dict(np.load(state_path))
    done, b_idx = int(st["done"]), int(st["b_idx"])
else:
    # ---- pilot batch: freeze bin edges (equal mass) + marginal histogram ranges
    rng = np.random.default_rng(7777)
    X = rng.standard_normal((60_000, n), dtype=np.float32)
    Z = X
    spikes = []
    for l in range(DEPTH):
        Z = Z @ Ms[l].T
        spikes.append(Z[:, 0].astype(np.float64).copy())
        Z = np.maximum(Z, 0)
    st = {}
    qs = np.linspace(0, 1, NBINS + 1)[1:-1]
    for l in (1, 2):            # 0-indexed layers with knees: pre-act layers 2 and 3
        e = np.quantile(spikes[l], qs)
        st[f"edges{l}"] = np.concatenate([[-np.inf], e, [np.inf]])
    for l in (0, 1, 2):
        m, sdev = spikes[l].mean(), spikes[l].std()
        st[f"hgrid{l}"] = np.linspace(m - 6 * sdev, m + 6 * sdev, HB + 1)
        st[f"hist{l}"] = np.zeros(HB)
    for l in (1, 2):
        st[f"cnt{l}"] = np.zeros((2, NBINS))
        st[f"sa{l}"] = np.zeros((2, NBINS))
        st[f"SZpre{l}"] = np.zeros((2, NBINS, n - 1))
        st[f"SZpost{l}"] = np.zeros((2, NBINS, n - 1))
    for l in (0, 1):            # xi channel l+1 -> l+2, binned by the DESTINATION spike bins
        st[f"sxi{l}"] = np.zeros(NBINS)
        st[f"sxi2{l}"] = np.zeros(NBINS)
        # unconditional channel moments: xi, xi^2, ReLU(src spike) A+, A+^2, xi*A+
        st[f"chan{l}"] = np.zeros(5)
    done, b_idx = 0, 0

rng = np.random.default_rng(31337)
rng.bit_generator.advance(b_idx * 5 * 10 ** 7)
t0 = time.time()
while done < N_MC and time.time() - t0 < BUDGET:
    b = min(BATCH, N_MC - done)
    X = rng.standard_normal((b, n), dtype=np.float32)
    Zs, Hs = [], []
    Z = X
    for l in range(DEPTH):
        Z = Z @ Ms[l].T
        Zs.append(Z)
        Z = np.maximum(Z, 0)
        Hs.append(Z)
    h = b_idx % 2
    for l in (0, 1, 2):
        sp = Zs[l][:, 0].astype(np.float64)
        g = st[f"hgrid{l}"]
        st[f"hist{l}"] += np.histogram(sp, bins=g)[0]
    for l in (1, 2):
        sp = Zs[l][:, 0].astype(np.float64)
        bi = np.clip(np.searchsorted(st[f"edges{l}"], sp) - 1, 0, NBINS - 1)
        st[f"cnt{l}"][h] += np.bincount(bi, minlength=NBINS)
        st[f"sa{l}"][h] += np.bincount(bi, weights=sp, minlength=NBINS)
        Zb = Zs[l][:, 1:].astype(np.float64)
        Hb = Hs[l][:, 1:].astype(np.float64)
        pre = st[f"SZpre{l}"]; post = st[f"SZpost{l}"]
        for k in range(NBINS):
            m = bi == k
            if m.any():
                pre[h, k] += Zb[m].sum(0)
                post[h, k] += Hb[m].sum(0)
        # channel xi into THIS layer's spike: A^l = c*ReLU(A^{l-1}) + xi, xi = w^T H^{l-1}_bulk
        xi = (Hs[l - 1][:, 1:] @ ws[l - 1]).astype(np.float64)
        st[f"sxi{l-1}"] += np.bincount(bi, weights=xi, minlength=NBINS)
        st[f"sxi2{l-1}"] += np.bincount(bi, weights=xi * xi, minlength=NBINS)
        Ap = Hs[l - 1][:, 0].astype(np.float64)
        st[f"chan{l-1}"] += np.array([xi.sum(), (xi * xi).sum(), Ap.sum(),
                                      (Ap * Ap).sum(), (xi * Ap).sum()])
    done += b; b_idx += 1

st["done"] = done; st["b_idx"] = b_idx
if done < N_MC:
    np.savez_compressed(state_path, **st)
    print(f"PARTIAL {done}/{N_MC} ({time.time()-t0:.0f}s) -- rerun to continue", flush=True)
else:
    st["cs"] = cs
    np.savez_compressed(out_path, **st)
    try:
        os.remove(state_path)
    except OSError:
        pass          # sandbox mounts may forbid unlink; stale .partial is ignored
    print("wrote", out_path, flush=True)

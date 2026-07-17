"""Channel-structure MC: what exactly does the 1-d spike recursion drop, and how fast
does its marginal prediction converge in width?

Exact identity per layer:  A_{l+1} = c_l ReLU(A_l) + xi_l,  xi_l = w_l^T ReLU(Z^l_bulk).
The recursion idealizes  xi_l | A_l ~ N(mu_l, om2_l)  independent of A_l.  This script
measures the dropped structure directly, binned by the SOURCE spike A_l:
    E[xi | A_l = a]      (should be constant; drops kappa*a = (w^T m1) a = O(1/sqrt n),
                          and the old-knee leak (w^T lambda) g_l(a) = O(1/sqrt n))
    Var[xi | A_l = a]    (should be constant; tilt O(1/sqrt n))
    skew[xi | A_l = a]   (should be 0; O(1/sqrt n) by transverse-cumulant power counting)
all with SPLIT HALVES so deviation powers can be estimated unbiasedly (cross products),
plus split-half fine histograms of the spike marginals A_1, A_2, A_3 for the
cross-half-debiased L2 density error of the recursion,
    D^2 = int (p_hat - p_rec)^2 da   via   sum da (p_A - p_rec)(p_B - p_rec).
Prediction: dropped-term amplitudes ~ n^{-1/2}  =>  D^2 ~ n^{-1}.

Usage:  python knee_channel_mc.py [n] [seed] [N_MC]     (resumable, BUDGET_S env)
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
DEPTH, NB, HB = 3, 24, 321
BATCH = min(60_000, max(15_000, 2 ** 24 // n))
BUDGET = float(os.environ.get("BUDGET_S", 27))

out_path = os.path.join(CKPT, f"chanmc_n{n}_seed{seed}_N{N_MC}.npz")
state_path = out_path + ".partial.npz"
if os.path.exists(out_path):
    print("already complete:", out_path); sys.exit(0)

Ws = build_spiked_net(n, DEPTH, seed=seed)
Ms = [W.astype(np.float32) for W, _ in Ws[:DEPTH]]
cs = np.array([Ws[l + 1][0][0, 0] for l in range(DEPTH - 1)])
ws = [Ws[l + 1][0][0, 1:].astype(np.float32) for l in range(DEPTH - 1)]

if os.path.exists(state_path):
    st = dict(np.load(state_path))
    done, b_idx = int(st["done"]), int(st["b_idx"])
else:
    rng = np.random.default_rng(7777)
    X = rng.standard_normal((60_000, n), dtype=np.float32)
    Z = X; spikes = []
    for l in range(DEPTH):
        Z = Z @ Ms[l].T
        spikes.append(Z[:, 0].astype(np.float64).copy())
        Z = np.maximum(Z, 0)
    st = {}
    qs = np.linspace(0, 1, NB + 1)[1:-1]
    for l in (0, 1):                      # SOURCE-spike bins for channels l -> l+1
        e = np.quantile(spikes[l], qs)
        st[f"sedges{l}"] = np.concatenate([[-np.inf], e, [np.inf]])
        for k in ("cnt", "sa", "sxi", "sxi2", "sxi3"):
            st[f"{k}{l}"] = np.zeros((2, NB))
        st[f"chan{l}"] = np.zeros(5)      # xi, xi^2, A+, A+^2, xi*A+
    for l in (0, 1, 2):
        m, s_ = spikes[l].mean(), spikes[l].std()
        st[f"hgrid{l}"] = np.linspace(m - 6 * s_, m + 6 * s_, HB + 1)
        st[f"hist{l}"] = np.zeros((2, HB))
    done, b_idx = 0, 0

rng = np.random.default_rng(97531)
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
        st[f"hist{l}"][h] += np.histogram(Zs[l][:, 0].astype(np.float64),
                                          bins=st[f"hgrid{l}"])[0]
    for l in (0, 1):
        a = Zs[l][:, 0].astype(np.float64)
        bi = np.clip(np.searchsorted(st[f"sedges{l}"], a) - 1, 0, NB - 1)
        xi = (Hs[l][:, 1:] @ ws[l]).astype(np.float64)
        st[f"cnt{l}"][h] += np.bincount(bi, minlength=NB)
        st[f"sa{l}"][h] += np.bincount(bi, weights=a, minlength=NB)
        st[f"sxi{l}"][h] += np.bincount(bi, weights=xi, minlength=NB)
        st[f"sxi2{l}"][h] += np.bincount(bi, weights=xi * xi, minlength=NB)
        st[f"sxi3{l}"][h] += np.bincount(bi, weights=xi ** 3, minlength=NB)
        Ap = Hs[l][:, 0].astype(np.float64)
        st[f"chan{l}"] += np.array([xi.sum(), (xi * xi).sum(), Ap.sum(),
                                    (Ap * Ap).sum(), (xi * Ap).sum()])
    done += b; b_idx += 1

st["done"] = done; st["b_idx"] = b_idx
if done < N_MC:
    np.savez_compressed(state_path, **st)
    print(f"PARTIAL {done}/{N_MC} ({time.time()-t0:.0f}s)", flush=True)
else:
    st["cs"] = cs
    np.savez_compressed(out_path, **st)
    try:
        os.remove(state_path)
    except OSError:
        pass
    print("wrote", out_path, flush=True)

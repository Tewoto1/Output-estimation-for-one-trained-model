"""MC verification of the layer-2 knee at n=2048: depth-2 spiked net, bin a'=(M'h)_0,
per-bin bulk pre-act means (all coords, split halves) + variance along the predicted
lambda direction and a random control direction."""
import os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
CKPT = os.path.join(REPO, "checkpoints", "affine_conditional_layer1")
from Mecha_preds.binned_kprop.empirical_structure import build_spiked_net

n, seed = 2048, 0
N_MC, BATCH, NBINS = 960_000, 60_000, 24
out_path = os.path.join(CKPT, f"knee2_mc_n{n}_seed{seed}_N{N_MC}_B{NBINS}.npz")

th = np.load(os.path.join(CKPT, f"knee2_theory_n{n}_seed{seed}.npz"))
ap_g, p_g = th["ap"], th["p_ap"]
cdf = np.cumsum(p_g); cdf /= cdf[-1]
edges = np.interp(np.linspace(0, 1, NBINS + 1)[1:-1], cdf, ap_g)
edges = np.concatenate([[-np.inf], edges, [np.inf]])
lam_u = th["lam_hat"] / np.linalg.norm(th["lam_hat"])
rng0 = np.random.default_rng(4242)
ctrl = rng0.standard_normal(n - 1); ctrl -= (ctrl @ lam_u) * lam_u; ctrl /= np.linalg.norm(ctrl)

Ws = build_spiked_net(n, depth=2, seed=seed)
M1 = Ws[0][0].astype(np.float32); M2 = Ws[1][0].astype(np.float32)

state_path = out_path + ".partial.npz"
if os.path.exists(state_path):
    st = np.load(state_path)
    cnt, sa, SZ = st["cnt"], st["sa"], st["SZ"]
    Sq, Sq2, Sc, Sc2 = st["Sq"], st["Sq2"], st["Sc"], st["Sc2"]
    done, b_idx = int(st["done"]), int(st["b_idx"])
else:
    cnt = np.zeros((2, NBINS)); sa = np.zeros((2, NBINS))
    SZ = np.zeros((2, NBINS, n - 1))
    Sq = np.zeros(NBINS); Sq2 = np.zeros(NBINS); Sc = np.zeros(NBINS); Sc2 = np.zeros(NBINS)
    done, b_idx = 0, 0

rng = np.random.default_rng(999)
rng.bit_generator.advance(b_idx * 10**7 * 5)   # decorrelate resumed streams cheaply
t0 = time.time()
while done < N_MC and time.time() - t0 < 28:
    b = min(BATCH, N_MC - done)
    X = rng.standard_normal((b, n), dtype=np.float32)
    H = np.maximum(X @ M1.T, 0)
    Zp = H @ M2.T
    apv = Zp[:, 0].astype(np.float64)
    Zb = Zp[:, 1:].astype(np.float64)
    bi = np.clip(np.searchsorted(edges, apv) - 1, 0, NBINS - 1)
    h = b_idx % 2
    cnt[h] += np.bincount(bi, minlength=NBINS)
    sa[h] += np.bincount(bi, weights=apv, minlength=NBINS)
    for k in range(NBINS):
        m = bi == k
        if m.any():
            SZ[h, k] += Zb[m].sum(0)
    q = Zb @ lam_u; cc = Zb @ ctrl
    Sq += np.bincount(bi, weights=q, minlength=NBINS)
    Sq2 += np.bincount(bi, weights=q * q, minlength=NBINS)
    Sc += np.bincount(bi, weights=cc, minlength=NBINS)
    Sc2 += np.bincount(bi, weights=cc * cc, minlength=NBINS)
    done += b; b_idx += 1
    print(f"{done}/{N_MC} ({time.time()-t0:.0f}s)", flush=True)

if done < N_MC:
    np.savez_compressed(state_path, cnt=cnt, sa=sa, SZ=SZ, Sq=Sq, Sq2=Sq2, Sc=Sc, Sc2=Sc2,
                        done=done, b_idx=b_idx)
    print(f"PARTIAL saved at {done}/{N_MC} -- rerun to continue")
else:
    np.savez_compressed(out_path, cnt=cnt, sa=sa, SZ=SZ, Sq=Sq, Sq2=Sq2, Sc=Sc, Sc2=Sc2,
                        edges=edges, lam_u=lam_u, ctrl=ctrl)
    if os.path.exists(state_path):
        os.remove(state_path)
    print("wrote", out_path)

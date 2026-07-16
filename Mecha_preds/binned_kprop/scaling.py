"""scaling.py -- the SCALING-LAW-WITH-WIDTH test for coordinate-spike binned kprop (K=2).

Verifies the cumulant-propagation budget law: a budget-``k_max`` predictor has output
error ``O(n^{-k_max})`` per the kprop paper, i.e. at K = 2 the (relative) MEAN-SQUARED
error of the predicted output decays like ``MSE ~ n^{-2}`` (equivalently RMS error
``~ n^{-1}``) as the hidden width ``n`` grows. Run:

    python -m Mecha_preds.binned_kprop.scaling

The model is the COORDINATE-spiked net ``M = W + e_1 e_1^T``. The point of the binning:
the spike coordinate carries O(1) cumulants at every order (no flat-loop discount), so a
naive Gaussian-in-the-spike closure -- here the ``num_bins = 1`` baseline, which collapses
coordinate 0 to its mean and eats a ReLU Jensen gap -- does NOT achieve the K=2 rate (its
MSE decays only ~ ``n^{-1}``). Representing the spike coordinate with enough bins removes
that O(1) error, so the remaining (bulk K=2 Gaussian-closure) error follows the paper rate
``MSE ~ n^{-2}``. The test measures, seed-averaged over random nets:

  * binned-K2 relative MSE vs width and its fitted log-log slope (target ~ -2);
  * the single-bin baseline slope (only ~ -1) for contrast;
  * a fixed-width bin-REFINEMENT curve (error vs num_bins).

NOTE you must use enough bins to see the n^{-2} rate -- with too few bins the spike
discretization error floors the total error and flattens the width slope. The default
``num_bins = 31`` is past that knee for the widths tested here.

Hard gates (robust, seed-averaged):
  (a) refinement: error drops sharply with num_bins, then converges to a small floor;
  (b) binning helps: binned MSE < single-bin MSE at every width, by a clear margin;
  (c) K=2 width law: binned MSE slope ~ -2 (gate ``< -1.5``) AND clearly steeper than the
      single-bin slope -- i.e. binning recovers the paper's ``O(n^{-k_max})`` rate.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .core import run_binned_kprop_k2
from .selftest import coordinate_spike_net, mc_output_mean, _rel


def fit_loglog_slope(widths: Sequence[int], errs: Sequence[float]) -> float:
    """Least-squares slope ``p`` of ``log(err) = p log(n) + c`` (so ``err ~ n^p``)."""
    x = np.log(np.asarray(widths, dtype=np.float64))
    y = np.log(np.asarray(errs, dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


def width_scaling(widths: Sequence[int] = (32, 64, 128), *, depth: int = 2,
                  num_bins: int = 31, seeds: Sequence[int] = (10, 11, 12, 13),
                  samples: int = 2_500_000, batch: int = 250_000, theta: float = 1.0,
                  out_dim: int = 8, bulk_relu: str = "exact", grid: str = "fixed") -> Dict[str, object]:
    """Seed-averaged binned-K2 vs single-bin error across widths, with fitted exponents.

    Metric: relative L2 error of the predicted output-mean VECTOR (``out_dim`` random
    readouts -> stable norm) vs Monte-Carlo. Returns RMS-error and MSE (= RMS^2) arrays
    plus their fitted log-log slopes; ``mse_slope_binned`` is the K=2 width law (~ -2).
    Also returns the worst per-width MC noise floor so you can tell you are not noise-limited.
    """
    widths = list(widths)
    rms_binned, rms_single, noise = [], [], []
    for n in widths:
        eb, es, nz = [], [], []
        for s in seeds:
            Ws = coordinate_spike_net(n, depth, seed=s, theta=theta, out_dim=out_dim)
            mc, se = mc_output_mean(Ws, n, samples, batch, seed=1000 + s)
            eb.append(_rel(run_binned_kprop_k2(Ws, n, num_bins=num_bins, grid=grid, bulk_relu=bulk_relu)["mean"], mc))
            # the naive single-Gaussian-spike baseline: PIN grid="fixed" (one all-reals bin;
            # the driver default is now the W2-adaptive grid, which would split at 0)
            es.append(_rel(run_binned_kprop_k2(Ws, n, num_bins=1, grid="fixed", bulk_relu=bulk_relu)["mean"], mc))
            nz.append(float(np.linalg.norm(se) / (np.linalg.norm(mc) + 1e-30)))
        rms_binned.append(float(np.mean(eb)))
        rms_single.append(float(np.mean(es)))
        noise.append(float(np.mean(nz)))
    rms_binned = np.array(rms_binned); rms_single = np.array(rms_single)
    mse_binned = rms_binned ** 2; mse_single = rms_single ** 2
    return {
        "widths": np.array(widths, dtype=float),
        "rms_binned": rms_binned, "rms_single": rms_single,
        "mse_binned": mse_binned, "mse_single": mse_single,
        "noise": np.array(noise),
        "ratio": rms_single / np.clip(rms_binned, 1e-30, None),
        "rms_slope_binned": fit_loglog_slope(widths, rms_binned),
        "rms_slope_single": fit_loglog_slope(widths, rms_single),
        "mse_slope_binned": fit_loglog_slope(widths, mse_binned),
        "mse_slope_single": fit_loglog_slope(widths, mse_single),
    }


def bin_refinement(n: int = 64, *, depth: int = 2, seeds: Sequence[int] = (10, 11, 12),
                   num_bins_list: Sequence[int] = (1, 3, 7, 15, 31),
                   samples: int = 1_500_000, batch: int = 250_000, theta: float = 1.0,
                   out_dim: int = 8, bulk_relu: str = "exact",
                   grid: str = "fixed") -> Dict[str, np.ndarray]:
    """Fixed-width error vs ``num_bins`` (seed-averaged): the refinement curve.

    ``grid="fixed"`` by default so the ``num_bins=1`` point stays the naive
    single-Gaussian-spike closure the gates were calibrated on."""
    errs = []
    for nb in num_bins_list:
        e = []
        for s in seeds:
            Ws = coordinate_spike_net(n, depth, seed=s, theta=theta, out_dim=out_dim)
            mc, _se = mc_output_mean(Ws, n, samples, batch, seed=2000 + s)
            e.append(_rel(run_binned_kprop_k2(Ws, n, num_bins=nb, grid=grid,
                                              bulk_relu=bulk_relu)["mean"], mc))
        errs.append(float(np.mean(e)))
    return {"num_bins": np.array(num_bins_list, dtype=float), "err": np.array(errs)}


def run(verbose: bool = True, *, quick: bool = False) -> bool:
    ok = True
    seeds = (10, 11, 12) if quick else (10, 11, 12, 13)
    samples = 1_500_000 if quick else 2_500_000

    # --- K=2 width law:  MSE ~ n^{-2} ---------------------------------------
    sweep = width_scaling(widths=(32, 64, 128), depth=2, num_bins=31,
                          seeds=seeds, samples=samples)
    w = sweep["widths"]
    rms_b, rms_s = sweep["rms_binned"], sweep["rms_single"]
    mse_b = sweep["mse_binned"]; rat = sweep["ratio"]; nz = sweep["noise"]
    mse_slope = sweep["mse_slope_binned"]; mse_slope_single = sweep["mse_slope_single"]

    helps_ok = bool(np.all(rms_b < rms_s) and np.all(rat > 3.0))               # (b)
    rate_ok = (mse_slope < -1.5) and (mse_slope < mse_slope_single - 0.5)      # (c) ~ -2 & beats naive
    not_noise_limited = bool(np.all(rms_b > 3.0 * nz))                          # measurement is meaningful
    ok &= helps_ok and rate_ok and not_noise_limited
    if verbose:
        print("=== K=2 width scaling law (coordinate spike e1, num_bins=31, depth=2) ===")
        print("   n      MSE(binned)   MSE(1 bin)   RMS adv   MC-noise floor")
        for i in range(len(w)):
            print(f"  {int(w[i]):4d}    {mse_b[i]:.3e}    {sweep['mse_single'][i]:.3e}   "
                  f"{rat[i]:6.1f}x    {nz[i]:.1e}")
        print(f"  fitted: binned  MSE ~ n^{mse_slope:+.2f}  (RMS ~ n^{sweep['rms_slope_binned']:+.2f})"
              f"   <- paper K=2 rate is n^-2")
        print(f"          1 bin   MSE ~ n^{mse_slope_single:+.2f}  (RMS ~ n^{sweep['rms_slope_single']:+.2f})"
              f"   <- naive Gaussian-spike closure, only ~ n^-1")
        print(f"  (b) binning helps at every width (>3x):           {'OK' if helps_ok else 'FAIL'}")
        print(f"  (c) binned MSE slope ~ -2 and beats naive by >0.5: {'OK' if rate_ok else 'FAIL'}")
        print(f"      (not MC-noise-limited, err>3x noise):          {'OK' if not_noise_limited else 'FAIL'}")

    # --- bin refinement at fixed width --------------------------------------
    # Error DROPS sharply with num_bins, then converges to the bulk-K2-closure FLOOR
    # (spec 17) -- it does not keep falling. Test "big early win + convergence", not
    # strict monotonicity (the post-knee wiggle is the closure floor + MC noise).
    refine = bin_refinement(n=64, depth=2, seeds=seeds,
                            num_bins_list=(1, 3, 7, 15, 31),
                            samples=(1_000_000 if quick else 1_500_000))
    nbins = refine["num_bins"]; rerr = refine["err"]
    floor = float(rerr.min())
    big_win = rerr[2] < 0.25 * rerr[0]            # 7 bins << 1 bin (refinement works)
    converged = rerr[-1] <= 1.5 * floor           # tail sits at the floor (no divergence)
    floor_small = floor < 0.02                     # the floor itself is small
    refine_ok = bool(big_win and converged and floor_small)        # (a)
    ok &= refine_ok
    if verbose:
        print("=== bin refinement at fixed width n=64 ===")
        print("  num_bins:  " + "  ".join(f"{int(b):>3d}" for b in nbins))
        print("  rel err :  " + "  ".join(f"{e:.1e}" for e in rerr))
        print(f"  (a) sharp refinement + converges to small floor {floor:.1e}:  "
              f"{'OK' if refine_ok else 'FAIL'}")

    print("SCALING TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)

"""benchmark.py -- practical num_bins / width TUNING harness for binned kprop (K=2).

Not a pass/fail test (that's ``scaling.py``) -- this is the hands-on tool for tuning
``num_bins`` at a given (larger) width in practice: run the predictor against a fast
Monte-Carlo reference and read off accuracy vs wall-time vs the MC noise floor, so you
can pick the smallest ``num_bins`` that's "good enough". Classic defaults: depth 3,
4,000,000 MC samples in batches of 200,000.

    # default sweep at a larger width
    python -m Mecha_preds.binned_kprop.benchmark

    # tune a specific case
    python -m Mecha_preds.binned_kprop.benchmark --widths 256 512 --num-bins 11 21 41 \
        --depth 3 --samples 4000000 --batch 200000 --bulk-relu exact

Columns: rel-err (vs MC), MC-z (= ||err|| / ||MC se||; >~3 means the gap is real, not MC
noise), MC-noise floor, mass-lost / psd-clip diagnostics, and predict wall-time. As width
grows the K=2 error shrinks ~1/n (MSE ~ n^-2), so at large width with too few samples the
predictor can dip BELOW the MC noise floor (MC-z -> small) -- then add MC samples, not bins.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Sequence

import numpy as np

from .core import run_binned_kprop_k2
from .selftest import coordinate_spike_net, mc_output_mean, _rel


def benchmark(widths: Sequence[int] = (256,), num_bins_list: Sequence[int] = (11, 21, 41),
              *, depth: int = 3, samples: int = 4_000_000, batch: int = 200_000,
              seed: int = 7, theta: float = 1.0, out_dim: int = 8,
              bulk_relu: str = "exact", verbose: bool = True) -> List[Dict[str, float]]:
    """Sweep ``num_bins`` (and widths) on a coordinate-spiked depth-``depth`` net; compare
    to MC. Returns a list of per-(width, num_bins) result dicts. The MC reference is
    computed ONCE per width and reused across the ``num_bins`` sweep."""
    rows: List[Dict[str, float]] = []
    for n in widths:
        Ws = coordinate_spike_net(n, depth, seed=seed, theta=theta, out_dim=out_dim)
        t0 = time.time()
        mc, se = mc_output_mean(Ws, n, samples, batch, seed=10_000 + seed)
        mc_t = time.time() - t0
        noise = float(np.linalg.norm(se) / (np.linalg.norm(mc) + 1e-30))
        if verbose:
            print(f"\n# width n={n}, depth={depth}, bulk_relu={bulk_relu}, "
                  f"MC={samples:,} ({mc_t:.1f}s, rel-noise {noise:.1e})")
            print("  num_bins   rel-err     MC-z   mass-lost   psd-clip   predict[s]")
        for nb in num_bins_list:
            t0 = time.time()
            res = run_binned_kprop_k2(Ws, n, num_bins=nb, bulk_relu=bulk_relu, collect=True)
            pt = time.time() - t0
            rel = _rel(res["mean"], mc)
            zmc = float(np.linalg.norm(res["mean"] - mc) / (np.linalg.norm(se) + 1e-30))
            md = res["metadata"]
            mass = max(md["max_linear_mass_lost"], md["max_relu_mass_lost"])
            rows.append(dict(width=n, num_bins=nb, rel_err=rel, mc_z=zmc, mc_noise=noise,
                             mass_lost=mass, psd_clip=md["total_psd_clipped"],
                             predict_s=pt, mc_s=mc_t))
            if verbose:
                print(f"  {nb:7d}   {rel:.3e}   {zmc:6.1f}   {mass:.1e}   "
                      f"{md['total_psd_clipped']:.1e}   {pt:8.2f}")
    if verbose and len(widths) >= 1:
        rec = recommend(rows)
        for n, nb in rec.items():
            print(f"# recommended num_bins @ n={n}: {nb} "
                  f"(smallest within 1.2x of the best rel-err at this width)")
    return rows


def recommend(rows: List[Dict[str, float]], *, slack: float = 1.2) -> Dict[int, int]:
    """For each width, the smallest ``num_bins`` whose rel-err is within ``slack`` x of the
    best rel-err seen at that width (the practical accuracy/cost knee)."""
    out: Dict[int, int] = {}
    widths = sorted({int(r["width"]) for r in rows})
    for n in widths:
        rs = sorted((r for r in rows if int(r["width"]) == n), key=lambda r: r["num_bins"])
        best = min(r["rel_err"] for r in rs)
        out[n] = int(next((r["num_bins"] for r in rs if r["rel_err"] <= slack * best),
                          rs[-1]["num_bins"]))
    return out


def _main(argv=None):
    ap = argparse.ArgumentParser(description="num_bins / width tuning for binned kprop (K=2)")
    ap.add_argument("--widths", type=int, nargs="+", default=[256])
    ap.add_argument("--num-bins", type=int, nargs="+", default=[11, 21, 41])
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--samples", type=int, default=4_000_000)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--out-dim", type=int, default=8)
    ap.add_argument("--bulk-relu", choices=["exact", "gain", "kprop"], default="exact")
    args = ap.parse_args(argv)
    benchmark(args.widths, args.num_bins, depth=args.depth, samples=args.samples,
              batch=args.batch, seed=args.seed, theta=args.theta, out_dim=args.out_dim,
              bulk_relu=args.bulk_relu, verbose=True)


if __name__ == "__main__":
    _main()

"""binning_scaling_experiment.py -- width-scaling law of the ANALYTIC AFFINE K=2
predictor (Mecha_preds.analytic_kprop) at A100 scale.

Replaces the old in-notebook scaling sections (the notebook keeps only cheap
validation/diagnostics). Measures, per node budget ``num_nodes`` and per width,
the relative MSE of E[model(X)] against a high-precision Monte-Carlo reference,
on random coordinate-spiked nets M = W + theta*e1 e1^T (identical builder/seeds
to the notebooks, so caches are shared where formats match). The binned companion
at a matched bin budget is included as the baseline up to ``--binned-max-width``.

WIDTHS default to the previous sweep PLUS 1536, 2048, 3072, 4096, with 10 seeds.
At these widths the plain rel-MSE hits the MC-noise floor, so the primary
estimator is the SPLIT-HALF CROSS-MSE (the widthlaw_significance trick): the MC
mean is accumulated in two independent halves (muA, muB) and

    cross_mse = <pred - muA, pred - muB> / ||mu||^2

is an UNBIASED estimate of ||pred - E[out]||^2/||mu||^2 -- the noise floor cancels
(individual estimates may be negative below the floor; seed means are reported
as-is and only positive means enter slope fits). Plain rel-MSE + the noise floor
are logged alongside.

PARALLELISM (A100 box): one thread budget ``--threads`` (default: all cores) is
split into OUTER seed-tasks x INNER per-node workers:
  * outer -- seeds run concurrently (ThreadPoolExecutor; the analytic core is
    numpy/scipy work that releases the GIL), capped by an estimated per-task
    memory footprint against ``--mem-gb`` at each width;
  * inner -- each run threads its per-node exact-ReLU loop (``workers=inner``);
    outer x inner ~= threads, so small widths get 10-way seed parallelism and
    the big widths shift the budget into per-run threading.
The GPU does the MC references (float64) and, for n >= ``--gpu-congr-min-width``,
the analytic Sigma-stack congruences (``device="cuda"``). Results are identical
regardless of threads/device (see analytic_kprop selftest [8]/[9]).

RUN on an A100 via ARC infra (single GPU; use --num-gpus 1 if the box has more):

    c launch a100 --num-gpus 1 --gpu-type 'A100*'
    c run a100 "experiments/analytic_kprop/binning_scaling_experiment.py" \
        --run-name analytic-binning-scaling
    c tail a100

Outputs go to $RESULTS_DIR (set by `c run`; defaults to results/analytic_kprop_scaling
locally): points.jsonl (per-point, incremental), results.csv / results.json
(aggregates + slopes), summary.log (human-readable table) -- all in the
ARC-infra auto-sync formats -- plus scaling.png. Collate later with
``arc_infra.analysis.get_results``.

RESUMABLE (the repo rule): MC references are cached as
checkpoints/analytic_kprop/mc2_*.npz (with halves; the notebooks' legacy mc_*.npz
lack halves and use different sample counts, so they are not reused) and every
prediction VECTOR is cached as checkpoints/analytic_kprop/pred/*.npz keyed by the
net + predictor config only -- re-running with a bigger MC budget re-scores the
cached predictions instead of recomputing them.

Local smoke test:   python experiments/analytic_kprop/binning_scaling_experiment.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# --------------------------------------------------------------------------- #
# repo bootstrap (script can be launched from anywhere, incl. `c run`)
# --------------------------------------------------------------------------- #
def _find_repo() -> str:
    starts = [os.environ.get("REPO_ROOT", "")]
    try:
        starts.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        # ARC-infra `c run` exec's the source with no __file__; REPO_ROOT is
        # exported by RUN_ENV in ~/.arc_infra_config.py
        pass
    starts += [os.getcwd(), os.path.expanduser("~/code/one_trained_case")]
    for start in starts:
        here = start
        for _ in range(6):
            if here and os.path.isfile(os.path.join(here, "model", "mlp.py")):
                return here
            here = os.path.dirname(here)
    raise RuntimeError("could not locate the repo root (model/mlp.py marker)")


REPO = _find_repo()
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from Mecha_preds.analytic_kprop import run_analytic_kprop_k2          # noqa: E402
from Mecha_preds.binned_kprop import run_binned_kprop_k2              # noqa: E402

CKPT_DIR = os.path.join(REPO, "checkpoints", "analytic_kprop")
PRED_DIR = os.path.join(CKPT_DIR, "pred")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--widths", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096])
    p.add_argument("--num-seeds", type=int, default=10, help="seeds 10..10+N-1 (default 10)")
    p.add_argument("--num-nodes", type=int, nargs="+", default=[6, 12, 20, 40, 80],
                   help="analytic node budgets to sweep")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--out-dim", type=int, default=8)
    p.add_argument("--grid", default="w2", choices=["w2", "uniform"])
    p.add_argument("--cov-intercept", default="mc", choices=["mc", "ls"])
    p.add_argument("--fit", default="post", choices=["pre", "post", "both"],
                   help="where the affine family is fitted: 'post' (cheap, O(d^2) "
                        "state memory -- recommended at the big widths), 'pre' "
                        "(paper Algorithm 7.2; ~num_nodes*d^2*8B state, heavy at "
                        "n>=3072), or 'both' to compare")
    p.add_argument("--atom", default="exact", choices=["exact", "fit"],
                   help="fit='post' only: keep the zero atom (merge of all negative "
                        "cells) as a separate EXACT component, or fold it into the "
                        "affine family (the linearity-includes-the-atom hypothesis)")
    p.add_argument("--mc-samples", type=int, default=20_000_000)
    p.add_argument("--mc-batch", type=int, default=200_000)
    p.add_argument("--binned-bins", type=int, default=40)
    p.add_argument("--binned-max-width", type=int, default=1024,
                   help="run the binned baseline up to this width (0 = skip)")
    p.add_argument("--threads", type=int, default=0, help="total thread budget (0 = all cores)")
    p.add_argument("--mem-gb", type=float, default=0.0,
                   help="RAM budget for concurrent seed-tasks (0 = 60%% of system RAM)")
    p.add_argument("--gpu-congr-min-width", type=int, default=1024,
                   help="use device='cuda' for the analytic congruences from this width up")
    p.add_argument("--no-gpu", action="store_true", help="force numpy/CPU everywhere")
    p.add_argument("--quick", action="store_true", help="tiny local smoke test")
    a = p.parse_args()
    if a.quick:
        a.widths = [16, 32, 64]
        a.num_seeds = 3
        a.num_nodes = [6, 12]
        a.mc_samples = 200_000
        a.binned_max_width = 64
    a.seeds = list(range(10, 10 + a.num_seeds))
    a.threads = a.threads or (os.cpu_count() or 4)
    if a.mem_gb <= 0:
        try:
            with open("/proc/meminfo") as f:
                total_kb = int(next(l for l in f if l.startswith("MemTotal")).split()[1])
            a.mem_gb = 0.6 * total_kb / 1e6
        except Exception:
            a.mem_gb = 16.0
    return a


def _gpu_available(args) -> bool:
    if args.no_gpu:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ModuleNotFoundError:
        return False


# --------------------------------------------------------------------------- #
# nets + Monte-Carlo with split halves
# --------------------------------------------------------------------------- #
def coordinate_spike_net(n, depth, seed, *, theta, out_dim):
    """IDENTICAL to the notebook builders (same seeds -> same nets)."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n)); P[0, 0] = theta
    Ws = [(rng.standard_normal((n, n)) / np.sqrt(n) + P, None) for _ in range(depth)]
    Ws.append((rng.standard_normal((out_dim, n)) / np.sqrt(n), None))
    return Ws


def _mc_halves_numpy(Ws, n, samples, batch, seed):
    rng = np.random.default_rng(seed)
    batch = min(batch, max(1, samples // 2))            # >= 2 batches so BOTH halves fill
    out_dim = Ws[-1][0].shape[0]
    acc = [np.zeros(out_dim), np.zeros(out_dim)]        # two independent halves
    cnt = [0, 0]
    accsq = np.zeros(out_dim); c = 0; k = 0
    while c < samples:
        b = min(batch, samples - c)
        h = rng.standard_normal((b, n))
        for li, (W, _b) in enumerate(Ws):
            z = h @ W.T
            h = np.maximum(z, 0.0) if li < len(Ws) - 1 else z
        acc[k % 2] += h.sum(0); cnt[k % 2] += b
        accsq += (h ** 2).sum(0); c += b; k += 1
    muA, muB = acc[0] / max(cnt[0], 1), acc[1] / max(cnt[1], 1)
    mu = (acc[0] + acc[1]) / c
    se = np.sqrt(np.clip(accsq / c - mu ** 2, 0, None) / c)
    return mu, se, muA, muB, cnt[0], cnt[1]


def _mc_halves_torch(Ws, n, samples, batch, seed):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False       # keep full fp precision
    dev = torch.device("cuda")
    dt = torch.float64
    batch = min(batch, max(20_000, int(4e9 / (n * 8))), max(1, samples // 2))
    # (memory cap on the activation buffers; >= 2 batches so BOTH halves fill)
    Wt = [torch.as_tensor(W, dtype=dt, device=dev) for W, _ in Ws]
    g = torch.Generator(device=dev).manual_seed(seed)
    out_dim = Ws[-1][0].shape[0]
    acc = [torch.zeros(out_dim, dtype=dt, device=dev) for _ in range(2)]
    cnt = [0, 0]
    accsq = torch.zeros(out_dim, dtype=dt, device=dev); c = 0; k = 0
    while c < samples:
        b = min(batch, samples - c)
        h = torch.randn(b, n, generator=g, dtype=dt, device=dev)
        for li, W in enumerate(Wt):
            z = h @ W.T
            h = torch.relu(z) if li < len(Wt) - 1 else z
        acc[k % 2] += h.sum(0); cnt[k % 2] += b
        accsq += (h ** 2).sum(0); c += b; k += 1
    muA = (acc[0] / max(cnt[0], 1)).cpu().numpy()
    muB = (acc[1] / max(cnt[1], 1)).cpu().numpy()
    mu = ((acc[0] + acc[1]) / c).cpu().numpy()
    se = torch.sqrt(torch.clamp(accsq / c - torch.as_tensor(mu, device=dev) ** 2, min=0) / c)
    return mu, se.cpu().numpy(), muA, muB, cnt[0], cnt[1]


def mc_reference(args, n, seed, use_gpu):
    """Cached MC mean + s.e. + independent HALF means (for split-half cross-MSE)."""
    key = (f"mc2_d{args.depth}_w{n}_seed{seed}_th{args.theta:g}"
           f"_od{args.out_dim}_s{args.mc_samples}.npz")
    path = os.path.join(CKPT_DIR, key)
    if os.path.exists(path):
        z = np.load(path)
        # validate the halves (guards against files from before the >=2-batch fix)
        if "cntA" in z.files and int(z["cntA"]) > 0 and int(z["cntB"]) > 0:
            return z["mu"], z["se"], z["muA"], z["muB"]
        print(f"  [mc] n={n} seed={seed}: cached file lacks valid halves -> recomputing", flush=True)
    Ws = coordinate_spike_net(n, args.depth, seed, theta=args.theta, out_dim=args.out_dim)
    t0 = time.time()
    if use_gpu:
        mu, se, muA, muB, cA, cB = _mc_halves_torch(Ws, n, args.mc_samples, args.mc_batch, 10_000 + seed)
    else:
        mu, se, muA, muB, cA, cB = _mc_halves_numpy(Ws, n, args.mc_samples, args.mc_batch, 10_000 + seed)
    np.savez(path, mu=mu, se=se, muA=muA, muB=muB, cntA=cA, cntB=cB)
    print(f"  [mc] n={n} seed={seed}: {args.mc_samples:,} samples in {time.time()-t0:.1f}s "
          f"({'gpu' if use_gpu else 'cpu'})", flush=True)
    return mu, se, muA, muB


# --------------------------------------------------------------------------- #
# cached predictions (independent of the MC budget -> reusable forever)
# --------------------------------------------------------------------------- #
def pred_cached(args, kind, n, seed, budget, *, inner_workers, device, fit="pre"):
    """Prediction vector for (net, predictor config), cached as npz. ``kind`` is
    "ana" (budget = num_nodes; ``fit`` selects the variant) or "bin" (= num_bins)."""
    if kind == "ana":
        tag = (f"pred-ana_d{args.depth}_w{n}_s{seed}_nn{budget}_{args.grid}"
               f"_{args.cov_intercept}_th{args.theta:g}_od{args.out_dim}")
        if fit == "post":                       # pre keeps the legacy tag (old caches)
            tag += f"_post-{args.atom}"
    else:
        tag = (f"pred-bin_d{args.depth}_w{n}_s{seed}_nb{budget}"
               f"_th{args.theta:g}_od{args.out_dim}")
    path = os.path.join(PRED_DIR, tag + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["pred"], float(z["runtime"]), True
    Ws = coordinate_spike_net(n, args.depth, seed, theta=args.theta, out_dim=args.out_dim)
    t0 = time.time()
    if kind == "ana":
        pred = run_analytic_kprop_k2(
            Ws, n, num_nodes=budget, grid=args.grid, bulk_relu="exact",
            cov_intercept=args.cov_intercept, fit=fit, atom=args.atom,
            workers=inner_workers, device=device)["mean"]
    else:
        pred = run_binned_kprop_k2(Ws, n, num_bins=budget, bulk_relu="exact",
                                   workers=inner_workers)["mean"]
    runtime = time.time() - t0
    np.savez(path, pred=pred, runtime=runtime)
    return pred, runtime, False


# --------------------------------------------------------------------------- #
# thread/memory planning
# --------------------------------------------------------------------------- #
def plan_parallelism(args, n):
    """(outer seed-tasks, inner per-node workers). Estimated per-task peak RAM:
    fit='pre' carries (num_nodes, d, d) node-covariance stacks (~3.5 x nn_max x
    n^2 x 8B + merge copies + per-thread ReLU temporaries); fit='post' carries
    only the affine family + slot accumulators + kernel temporaries (~15 x n^2 x
    8B). 'both' plans for the heavier pre."""
    nn_max = max(args.num_nodes)
    if args.fit == "post":
        est_gb = (15 * n * n * 8) / 1e9 + (3 * n * n * 8) / 1e9 + 0.3
    else:
        est_gb = (3.5 * nn_max * n * n * 8) / 1e9 + (3 * n * n * 8) / 1e9 + 0.3
    outer = max(1, min(args.num_seeds, args.threads, int(args.mem_gb / est_gb)))
    inner = max(1, args.threads // outer)
    return outer, inner, est_gb


# --------------------------------------------------------------------------- #
# main sweep
# --------------------------------------------------------------------------- #
def relnorm2(v, mu):
    return float(v @ v) / float(mu @ mu)


def main():
    args = parse_args()
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))
    if world > 1 and rank != 0:
        print(f"rank {rank}: single-GPU script, exiting (launch with --num-gpus 1)")
        return
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    results_dir = os.environ.get("RESULTS_DIR",
                                 os.path.join(REPO, "results", "analytic_kprop_scaling"))
    os.makedirs(results_dir, exist_ok=True)
    points_path = os.path.join(results_dir, "points.jsonl")

    use_gpu = _gpu_available(args)
    print(f"widths={args.widths} seeds={args.seeds} num_nodes={args.num_nodes} "
          f"depth={args.depth} MC={args.mc_samples:,} gpu={use_gpu} "
          f"threads={args.threads} mem_gb={args.mem_gb:.0f}", flush=True)
    print(f"RESULTS_DIR={results_dir}", flush=True)

    all_points = []

    def emit(point):
        all_points.append(point)
        with open(points_path, "a") as f:
            f.write(json.dumps(point) + "\n")

    for n in args.widths:
        outer, inner, est_gb = plan_parallelism(args, n)
        ana_dev = ("cuda" if (use_gpu and n >= args.gpu_congr_min_width) else None)

        # phase A -- MC references (GPU-serial; cached)
        mcs = {s: mc_reference(args, n, s, use_gpu) for s in args.seeds}

        # phase B -- all seeds in parallel; each task sweeps the node budgets
        fits = ["pre", "post"] if args.fit == "both" else [args.fit]

        def seed_task(s):
            rows = []
            mu, se, muA, muB = mcs[s]
            mu2 = float(mu @ mu)
            noise_rel2 = float(se @ se) / mu2
            budgets = [("ana", nn, f) for nn in args.num_nodes for f in fits]
            if args.binned_max_width and n <= args.binned_max_width:
                budgets.append(("bin", args.binned_bins, None))
            for kind, budget, f in budgets:
                pred, runtime, cached = pred_cached(args, kind, n, s, budget,
                                                    inner_workers=inner, device=ana_dev,
                                                    fit=(f or "pre"))
                rows.append({
                    "kind": kind, "width": n, "seed": s, "budget": budget, "fit": f,
                    "rel_err": float(np.linalg.norm(pred - mu)) / float(np.linalg.norm(mu)),
                    "rel_mse": relnorm2(pred - mu, mu),
                    "cross_mse_rel": float((pred - muA) @ (pred - muB)) / mu2,
                    "noise_rel2": noise_rel2,
                    "runtime_s": runtime, "cached": bool(cached),
                })
            return rows

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=outer) as ex:
            for rows in ex.map(seed_task, args.seeds):
                for r in rows:
                    emit(r)
        print(f"[width {n:5d}] done in {time.time()-t0:6.1f}s  "
              f"(outer={outer} seed-tasks x inner={inner} workers, "
              f"~{est_gb:.1f}GB/task, analytic device={ana_dev or 'numpy'})", flush=True)

    # ---------------- aggregation ----------------
    def agg(kind, budget, fit=None):
        rows = []
        for n in args.widths:
            pts = [p for p in all_points
                   if p["kind"] == kind and p["width"] == n and p["budget"] == budget
                   and p.get("fit") == fit]
            if not pts:
                continue
            cross = np.array([p["cross_mse_rel"] for p in pts])
            plain = np.array([p["rel_mse"] for p in pts])
            rows.append({
                "kind": kind, "budget": budget, "fit": fit, "width": n, "n_seeds": len(pts),
                "cross_mse_mean": float(cross.mean()),
                "cross_mse_sem": float(cross.std(ddof=1) / np.sqrt(len(cross))) if len(cross) > 1 else 0.0,
                "plain_mse_mean": float(plain.mean()),
                "noise_rel2": float(np.mean([p["noise_rel2"] for p in pts])),
                "runtime_mean_s": float(np.mean([p["runtime_s"] for p in pts])),
            })
        return rows

    def fit_slope(rows, key, mask_fn):
        xs = [r["width"] for r in rows if mask_fn(r)]
        ys = [r[key] for r in rows if mask_fn(r)]
        if len(xs) < 2:
            return float("nan")
        return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])

    fits = ["pre", "post"] if args.fit == "both" else [args.fit]
    combos = [("ana", nn, f) for nn in args.num_nodes for f in fits]
    if args.binned_max_width:
        combos.append(("bin", args.binned_bins, None))
    aggregates = {}
    lines = [f"analytic_kprop binning scaling -- depth {args.depth}, seeds {args.seeds}, "
             f"MC {args.mc_samples:,}, fit={args.fit}"
             + (f" (atom={args.atom})" if args.fit != "pre" else ""), ""]
    lines.append("  kind      budget |" + "".join(f"  n={n:<9d}" for n in args.widths)
                 + "  slope(cross)  slope(plain>4xnoise)")
    for kind, budget, f in combos:
        rows = agg(kind, budget, f)
        aggregates[f"{kind}{('-' + f) if f else ''}@{budget}"] = rows
        s_cross = fit_slope(rows, "cross_mse_mean", lambda r: r["cross_mse_mean"] > 0)
        s_plain = fit_slope(rows, "plain_mse_mean",
                            lambda r: r["plain_mse_mean"] > 4 * r["noise_rel2"])
        by_w = {r["width"]: r for r in rows}
        cells = "".join(f"  {by_w[n]['cross_mse_mean']:+.2e}" if n in by_w else "  " + " " * 9
                        for n in args.widths)
        label = f"{kind}{('-' + f) if f else ''}"
        lines.append(f"  {label:9s} {budget:5d} |{cells}   n^{s_cross:+.2f}       n^{s_plain:+.2f}")
    lines.append("")
    lines.append("cells = seed-mean split-half cross-MSE (unbiased; negative = below the MC floor)")
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)

    with open(os.path.join(results_dir, "summary.log"), "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items()},
                   "aggregates": aggregates, "points": all_points}, f, indent=1)
    import csv
    with open(os.path.join(results_dir, "results.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["kind", "budget", "fit", "width", "n_seeds",
                                             "cross_mse_mean", "cross_mse_sem",
                                             "plain_mse_mean", "noise_rel2", "runtime_mean_s"])
        wcsv.writeheader()
        for rows in aggregates.values():
            for r in rows:
                wcsv.writerow(r)

    # ---------------- plot (best-effort) ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        for kind, budget, f in combos:
            key = f"{kind}{('-' + f) if f else ''}@{budget}"
            rows = aggregates[key]
            xs = [r["width"] for r in rows if r["cross_mse_mean"] > 0]
            ys = [r["cross_mse_mean"] for r in rows if r["cross_mse_mean"] > 0]
            style = "s--" if kind == "bin" else ("o-" if f != "pre" else "o:")
            plt.loglog(xs, ys, style, label=key)
        k0 = f"{combos[0][0]}{('-' + combos[0][2]) if combos[0][2] else ''}@{combos[0][1]}"
        rows0 = aggregates[k0]
        if rows0:
            w0, y0 = rows0[0]["width"], max(rows0[0]["cross_mse_mean"], 1e-12)
            ww = np.array(args.widths, float)
            plt.loglog(ww, y0 * (ww / w0) ** -2.0, "k:", alpha=.5, label="n^-2 (K=2)")
            plt.loglog([r["width"] for r in rows0], [r["noise_rel2"] for r in rows0],
                       ":", c="gray", label="MC-noise floor (plain)")
        plt.xlabel("width n"); plt.ylabel("relative MSE (split-half cross)")
        plt.title(f"analytic K=2 width scaling (depth {args.depth}, {len(args.seeds)} seeds)")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "scaling.png"), dpi=150)
        print(f"plot -> {os.path.join(results_dir, 'scaling.png')}", flush=True)
    except Exception as e:  # matplotlib genuinely optional on a bare instance
        print(f"(no plot: {e})", flush=True)


if __name__ == "__main__":
    main()

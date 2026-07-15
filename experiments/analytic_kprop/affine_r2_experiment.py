"""affine_r2_experiment.py -- how linear is the bulk-given-spike law, per layer,
and how does it scale with width? (GPU/parallel version; replaces the old
affine_r2 notebook.)

Both analytic variants compress the bulk conditional to an AFFINE family in the
spike coordinate; they differ in where the projection happens. This script
measures the hypothesis DIRECTLY, at every layer, separately for:

  * PRE-activation:  exact reconditioned cell moments  (m_hat_j, S_hat_j) vs y_j
  * POST-activation: exact post-ReLU node moments      (r_k, R_k)         vs a_k
    (and optionally WITH the zero atom as an a=0 data point -- the atom="fit"
    hypothesis: is the merge of all negative cells on the same line?)

by fitting the weighted linear model from the bins/nodes to the mean and the
covariance and reporting the weighted pooled R^2

    R^2 = 1 - SS_res / SS_tot        (Frobenius pooling for covariances)

plus var_scale in [0,1] = SS_tot / (weighted 2nd moment): when ~0 the target is
constant across bins and R^2 is meaningless (canonical case: layer-0 pre-cov,
exactly constant in the continuous limit). The propagated state is the exact
paper path (fit="pre" carries exact NONLINEAR node moments), so the measured
data is never contaminated by a projection of the kind being tested, except
across layers where the algorithm itself projects.

NO Monte-Carlo anywhere -- the question is internal to the surrogate -- so the
whole sweep is cheap; the heavy parts are the per-layer exact ReLU kernel and
the per-cell congruences inside ``percell_bulk_moments`` (torch-offloaded to
the GPU for n >= --gpu-min-width).

PARALLELISM: one thread budget (``--threads``, default all cores) split into
OUTER seed-tasks x INNER per-node workers, capped per width by an estimated
task memory footprint against ``--mem-gb`` (the pre path carries a
(num_nodes, d, d) node stack + the dense per-cell moments).

Mode AUTO-DETECTS like the repo notebooks (experiments.py: QUICK = CPU-only):
full sweep on a GPU box, quick smoke test otherwise; --quick / --full override.
RUN on a GPU box via ARC infra:

    c run [name] "experiments/analytic_kprop/affine_r2_experiment.py" \
        --run-name affine-r2
    c tail [name]

Outputs to $RESULTS_DIR (auto-sync formats): points.jsonl (one line per
width/seed/layer), results.csv + results.json (aggregates + slopes),
summary.log, r2_scaling.png (four panels: pre/post x mean/cov, 1-R^2 vs n,
one curve per layer). RESUMABLE: per-(width, seed) R^2 rows are cached as
checkpoints/analytic_kprop/r2/*.json (same keys the old notebook used).
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
# repo bootstrap
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

from Mecha_preds.analytic_kprop import (                                # noqa: E402
    gaussian_input_state, analytic_layer_update, percell_bulk_moments)
from Mecha_preds.analytic_kprop.core import (                           # noqa: E402
    _layer_block, _component_params, _pair_stats, _torch_device)

CKPT_DIR = os.path.join(REPO, "checkpoints", "analytic_kprop")
R2_DIR = os.path.join(CKPT_DIR, "r2")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _cuda_available() -> bool:
    """The repo-wide GPU signal (mirrors experiments.py: QUICK = CPU-only).
    Import ``experiments`` when possible (single source of truth); otherwise
    probe torch directly; no torch -> CPU-only."""
    try:
        import experiments as E
        return not E.QUICK
    except Exception:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False


def _detect_cpus() -> int:
    """The CPUs this process may actually use: cgroup/affinity-aware (a container
    or scheduler mask can differ from os.cpu_count(), in either direction)."""
    try:
        return len(os.sched_getaffinity(0)) or (os.cpu_count() or 4)
    except AttributeError:
        return os.cpu_count() or 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--widths", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512, 768, 1024])
    p.add_argument("--num-seeds", type=int, default=8, help="seeds 10..10+N-1")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--num-nodes", type=int, default=40)
    p.add_argument("--grid", default="w2", choices=["w2", "uniform"])
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--no-atom-fit", action="store_true",
                   help="skip the post-with-atom-included measurement")
    p.add_argument("--threads", type=int, default=0, help="total thread budget (0 = all cores)")
    p.add_argument("--mem-gb", type=float, default=0.0, help="0 = 60%% of system RAM")
    p.add_argument("--gpu-min-width", type=int, default=512,
                   help="torch-offload the per-cell congruences from this width up")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--quick", action="store_true", help="force the smoke-test sweep")
    p.add_argument("--full", action="store_true",
                   help="force the full sweep even on a CPU-only machine")
    a = p.parse_args()
    a.gpu_available = (not a.no_gpu) and _cuda_available()
    # FULL by default. (The old auto-quick-when-no-GPU heuristic misfired: a
    # CPU-only box is not necessarily a small box -- e.g. a 24-core host. Pass
    # --quick explicitly for the smoke test; --full is kept as a no-op.)
    if a.quick:
        a.widths = [16, 32, 64, 128]
        a.num_seeds = 3
    a.seeds = list(range(10, 10 + a.num_seeds))
    a.threads = a.threads or _detect_cpus()
    if a.mem_gb <= 0:
        try:
            with open("/proc/meminfo") as f:
                total_kb = int(next(l for l in f if l.startswith("MemTotal")).split()[1])
            a.mem_gb = 0.6 * total_kb / 1e6
        except Exception:
            a.mem_gb = 16.0
    return a


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def coordinate_spike_net(n, depth, seed, *, theta):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, n)); P[0, 0] = theta
    Ws = [(rng.standard_normal((n, n)) / np.sqrt(n) + P, None) for _ in range(depth)]
    Ws.append((rng.standard_normal((8, n)) / np.sqrt(n), None))
    return Ws


def wls_affine_r2(w, x, data):
    """Weighted affine fit data_j ~ b0 + b1 x_j; pooled R^2 (Frobenius for
    matrices) + var_scale in [0,1] = SS_tot / weighted second moment."""
    w = np.asarray(w, float); w = w / w.sum()
    x = np.asarray(x, float)
    D = np.asarray(data, float).reshape(len(w), -1)
    xbar = float(w @ x); vx = float(w @ (x - xbar) ** 2)
    Dbar = w @ D
    ss_tot = float(w @ ((D - Dbar) ** 2).sum(axis=1))
    scale = float(w @ (D ** 2).sum(axis=1)) + 1e-300
    if vx <= 1e-14 or ss_tot <= 1e-14 * scale:
        return float("nan"), ss_tot / scale
    b1 = (w * (x - xbar)) @ D / vx
    b0 = Dbar - b1 * xbar
    resid = D - b0[None, :] - x[:, None] * b1[None, :]
    ss_res = float(w @ (resid ** 2).sum(axis=1))
    return 1.0 - ss_res / ss_tot, ss_tot / scale


def affine_r2_by_layer(args, Ws, n, *, inner_workers, dev, tag=None):
    """Per-layer weighted R^2 of the affine hypothesis, pre and post.
    ``tag`` (e.g. "n=1024 seed=12") turns on per-layer progress prints -- the
    exact ReLU kernel + per-cell congruences make big widths take minutes."""
    d = n - 1
    state = gaussian_input_state(d)
    rows = []
    for li in range(len(Ws) - 1):
        t_layer = time.time()
        M = np.asarray(Ws[li][0], dtype=np.float64)
        gamma, r, u, V, beta, eta_b = _layer_block(M, None, d)
        mY, sY2, mC, g = _component_params(state, gamma, r, u, V, beta, eta_b)
        new_state, aff = analytic_layer_update(
            state, M, None, num_nodes=args.num_nodes, grid=args.grid,
            workers=inner_workers, dev=dev)
        # PRE: exact cell moments on the SAME retained grid the layer used
        Q, ym, dl, vv, stoch = _pair_stats(state.p, mY, sY2, aff.edges, min_prob=1e-15)
        wr = state.p @ Q; keep = wr > 1e-15
        mh, Sh = percell_bulk_moments(state.p, Q[:, keep], dl[:, keep], vv[:, keep],
                                      sY2, stoch, wr[keep],
                                      state.Sigma, state.t2, mC, g, u, V, dev)
        r2_pm, vs_pm = wls_affine_r2(aff.w, aff.y, mh)
        r2_pc, vs_pc = wls_affine_r2(aff.w, aff.y, Sh)
        # POST: exact post-ReLU node moments of the advanced state
        pos = new_state.a > 0
        r2_qm, vs_qm = wls_affine_r2(new_state.p[pos], new_state.a[pos], new_state.mu[pos])
        r2_qc, vs_qc = wls_affine_r2(new_state.p[pos], new_state.a[pos], new_state.Sigma[pos])
        row = dict(layer=li,
                   pre_mean=r2_pm, pre_cov=r2_pc, post_mean=r2_qm, post_cov=r2_qc,
                   vs_pre_mean=vs_pm, vs_pre_cov=vs_pc,
                   vs_post_mean=vs_qm, vs_post_cov=vs_qc)
        if (not args.no_atom_fit) and (~pos).any():
            r2_am, _ = wls_affine_r2(new_state.p, np.maximum(new_state.a, 0.0), new_state.mu)
            r2_ac, _ = wls_affine_r2(new_state.p, np.maximum(new_state.a, 0.0), new_state.Sigma)
            row.update(post_mean_atom=r2_am, post_cov_atom=r2_ac)
        rows.append(row)
        state = new_state
    return rows


def r2_cached(args, n, seed, *, inner_workers, dev):
    path = os.path.join(R2_DIR, f"r2_d{args.depth}_w{n}_s{seed}"
                                f"_nn{args.num_nodes}_{args.grid}.json")
    if os.path.exists(path):
        return json.load(open(path)), True
    t0 = time.time()
    Ws = coordinate_spike_net(n, args.depth, seed, theta=args.theta)
    rows = affine_r2_by_layer(args, Ws, n, inner_workers=inner_workers, dev=dev)
    json.dump(rows, open(path, "w"))
    print(f"  [r2] n={n} seed={seed}: {time.time()-t0:.1f}s", flush=True)
    return rows, False


def plan_parallelism(args, n):
    """Per-task peak RAM: the pre-path node stack + the dense per-cell moments
    (each ~num_nodes x d^2 x 8B) + kernel temporaries + the net."""
    est_gb = (2.8 * args.num_nodes * n * n * 8) / 1e9 + (3 * n * n * 8) / 1e9 + 0.3
    outer = max(1, min(args.num_seeds, args.threads, int(args.mem_gb / est_gb)))
    inner = max(1, args.threads // outer)
    return outer, inner, est_gb


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
PANELS = [("pre_mean", "pre-activation mean"), ("pre_cov", "pre-activation cov"),
          ("post_mean", "post-activation mean"), ("post_cov", "post-activation cov")]


def main():
    args = parse_args()
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1")))
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0")))
    if world > 1 and rank != 0:
        print(f"rank {rank}: single-process script, exiting (launch with --num-gpus 1)")
        return
    os.makedirs(R2_DIR, exist_ok=True)
    results_dir = os.environ.get("RESULTS_DIR",
                                 os.path.join(REPO, "results", "analytic_kprop_affine_r2"))
    os.makedirs(results_dir, exist_ok=True)
    use_gpu = args.gpu_available
    print(f"mode={'QUICK (no GPU detected; pass --full to override)' if args.quick else 'FULL'} "
          f"| gpu={use_gpu}")
    print(f"widths={args.widths} seeds={args.seeds} depth={args.depth} "
          f"num_nodes={args.num_nodes} threads={args.threads} mem_gb={args.mem_gb:.0f}")
    print(f"RESULTS_DIR={results_dir}", flush=True)

    all_rows = []                                          # (width, seed, layer-row)
    points_path = os.path.join(results_dir, "points.jsonl")
    for n in args.widths:
        outer, inner, est_gb = plan_parallelism(args, n)
        dev = (_torch_device("cuda") if (use_gpu and n >= args.gpu_min_width) else None)
        t0 = time.time()

        def seed_task(s):
            return s, r2_cached(args, n, s, inner_workers=inner, dev=dev)

        with ThreadPoolExecutor(max_workers=outer) as ex:
            for s, (rows, cached) in ex.map(seed_task, args.seeds):
                for row in rows:
                    pt = dict(width=n, seed=s, **row)
                    all_rows.append(pt)
                    with open(points_path, "a") as f:
                        f.write(json.dumps(pt) + "\n")
        print(f"[width {n:5d}] done in {time.time()-t0:6.1f}s  (outer={outer} x inner={inner}, "
              f"~{est_gb:.1f}GB/task, device={'cuda' if dev is not None else 'numpy'})", flush=True)

    # ---------------- aggregation: 1 - R^2 per (metric, layer, width) ----------------
    metrics = [c for c, _ in PANELS]
    if not args.no_atom_fit:
        metrics += ["post_mean_atom", "post_cov_atom"]

    def one_minus_r2(col, n, L):
        vals = [1.0 - p[col] for p in all_rows
                if p["width"] == n and p["layer"] == L
                and p.get(col) is not None and np.isfinite(p.get(col, np.nan))]
        return (float(np.mean(vals)),
                float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                len(vals))

    def slope(xs, ys):
        m = np.isfinite(ys) & (ys > 0)
        if m.sum() < 2:
            return float("nan")
        return float(np.polyfit(np.log(np.asarray(xs, float)[m]), np.log(ys[m]), 1)[0])

    agg = []
    lines = [f"affine-hypothesis R^2 scaling -- depth {args.depth}, seeds {args.seeds}, "
             f"num_nodes {args.num_nodes}", "",
             "1 - R^2 (seed mean); slope fitted over width per (metric, layer)", ""]
    for col in metrics:
        for L in range(args.depth):
            ys = []
            for n in args.widths:
                m_, sem, k = one_minus_r2(col, n, L)
                agg.append(dict(metric=col, layer=L, width=n,
                                one_minus_r2_mean=m_, sem=sem, n_seeds=k))
                ys.append(m_)
            ys = np.array(ys)
            sl = slope(args.widths, ys)
            cells = "  ".join(f"{v:.3e}" if np.isfinite(v) else "   --    " for v in ys)
            lines.append(f"  {col:15s} L{L} |  {cells}   ~ n^{sl:+.2f}")
        lines.append("")
    lines.insert(4, "  metric          L  |  " + "  ".join(f"n={n:<7d}" for n in args.widths))
    summary = "\n".join(lines)
    print("\n" + summary, flush=True)

    with open(os.path.join(results_dir, "summary.log"), "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "aggregates": agg, "points": all_rows}, f, indent=1)
    import csv
    with open(os.path.join(results_dir, "results.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["metric", "layer", "width",
                                             "one_minus_r2_mean", "sem", "n_seeds"])
        wcsv.writeheader()
        for r in agg:
            wcsv.writerow(r)

    # ---------------- plot ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        w = np.array(args.widths, float)
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
        for ax, (col, title) in zip(axes.ravel(), PANELS):
            for L in range(args.depth):
                ys = np.array([next(a["one_minus_r2_mean"] for a in agg
                                    if a["metric"] == col and a["layer"] == L
                                    and a["width"] == n) for n in args.widths])
                sl = slope(args.widths, ys)
                ax.loglog(w, ys, "o-", label=f"layer {L} ~ n^{sl:+.2f}")
                if col.startswith("post") and not args.no_atom_fit:
                    ya = np.array([next(a["one_minus_r2_mean"] for a in agg
                                        if a["metric"] == col + "_atom" and a["layer"] == L
                                        and a["width"] == n) for n in args.widths])
                    ax.loglog(w, ya, "--", alpha=.5, color=ax.lines[-1].get_color(),
                              label=("+atom" if L == 0 else None))
            ax.set_title(title + "  (1 - R^2)", fontsize=11)
            ax.set_xlabel("width n"); ax.legend(fontsize=7)
        plt.suptitle(f"affine-hypothesis error vs width (depth {args.depth}, "
                     f"{len(args.seeds)} seeds, {args.num_nodes} nodes)", y=1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "r2_scaling.png"), dpi=150)
        print(f"plot -> {os.path.join(results_dir, 'r2_scaling.png')}", flush=True)
    except Exception as e:
        print(f"(no plot: {e})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc(file=sys.stdout)   # `c tail` follows stdout
        sys.stdout.flush()
        raise

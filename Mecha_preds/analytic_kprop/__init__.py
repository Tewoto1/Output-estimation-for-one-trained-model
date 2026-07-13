"""analytic_kprop -- ANALYTIC AFFINE-CONDITIONED K=2 propagation (coordinate spike).

The 9th mechanistic predictor: implements *"Analytic Affine-Conditioned K=2
Propagation for Coordinate-Spiked ReLU Networks"* (analytic_affine_kprop.pdf,
Algorithm 7.2 with the exact-cell scalar backend). Same model class as the
companion ``..binned_kprop`` -- hidden matrices ``M = W + e_1 e_1^T`` -- but the
bulk conditional law between layers is compressed to ONE affine family

    C | Y = y  ~~>  N(mu0 + mu1 y,  Sigma0 + Sigma1 y),

instead of one bulk Gaussian per spike bin. The spike direction is discretized
only transiently, into ``num_nodes`` quadrature cells per layer whose
probabilities and truncated moments are closed-form ("bashed") from the KNOWN
Gaussian-mixture scalar law -- no per-node bulk state is propagated, so the
per-layer d^3 congruence cost is O(1) rather than O(num_bins):

    from model import MLP
    from Mecha_preds.analytic_kprop import run_analytic_kprop
    model, _ = MLP.load("checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt")
    pred = run_analytic_kprop(model, config={"num_nodes": 40})["mean"]

The core (``core.py``) is numpy/scipy and torch-free; only the ``adapter`` model
path touches torch. Scalar-law variants beyond the exact-cell backend (paper
section 7.3: mixture-integral, atomic-node, single-Gaussian) are future knobs.
"""
from .core import (
    SPIKE_COORD,
    AnalyticState,
    AffineState,
    gaussian_input_state,
    analytic_layer_update,
    negative_mass,
    split_node_budget,
    make_cells,
    percell_bulk_moments,
    unconditional_mean,
    unconditional_mean_cov,
    run_analytic_kprop_k2,
)
from .adapter import (
    run_analytic_kprop,
    default_analytic_kprop_config,
    config_summary,
    extract_mean,
)

__all__ = [
    # core
    "SPIKE_COORD", "AnalyticState", "AffineState", "gaussian_input_state",
    "analytic_layer_update", "negative_mass", "split_node_budget", "make_cells",
    "percell_bulk_moments", "unconditional_mean", "unconditional_mean_cov",
    "run_analytic_kprop_k2",
    # MLP adapter
    "run_analytic_kprop", "default_analytic_kprop_config", "config_summary",
    "extract_mean",
]

"""binned_kprop -- coordinate-spike BINNED cumulant propagation (K=2).

A mechanistic predictor for a ReLU MLP whose hidden matrices carry a *coordinate*
spike ``M = W + e_1 e_1^T`` (the spike sits on a single axis, not the flat all-ones
direction). For a flat spike, ordinary total-order kprop already works
(``..cumulants.spikekprop`` "ones" case). A coordinate spike has no flat-loop discount,
so the cumulants on coordinate 0 are O(1) at every order and cannot be carried as
ordinary bulk cumulant-tensor entries. This package instead runs a hidden-Markov-model
over the spike coordinate (a discrete distribution over ``num_bins`` bins, propagated by
a scalar transition kernel) with ordinary K=2 cumulant propagation of the bulk
*conditional on each bin*:

    from model import MLP
    from Mecha_preds.binned_kprop import run_binned_kprop
    model, _ = MLP.load("checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt")
    pred = run_binned_kprop(model, config={"num_bins": 41})["mean"]

THIS IS THE K = 2 IMPLEMENTATION (each bin stores a conditional bulk mean + covariance);
``num_bins`` is the adjustable hyperparameter. The general ``K > 2`` extension hooks into
ORDINARY harmonic kprop -- ``kprop_hook`` imports and calls
``Mecha_preds.cumulants.kprop`` for the per-bin bulk ReLU (``bulk_relu_kprop`` is also
selectable as the K=2 bulk-ReLU backend via ``bulk_relu="kprop"``).

The K=2 core (``core.py``) is numpy/scipy and torch-free; the ``kprop_hook`` /
``adapter`` paths use torch.
"""
from .core import (
    BinnedK2State,
    normal_interval_stats,
    find_bin,
    safe_bin_representative,
    symmetrize,
    project_to_psd,
    make_gaussian_edges,
    make_relu_post_edges,
    lloyd_max_edges,
    gaussian_initial_state,
    linear_step_k2,
    relu_step_k2,
    unconditional_mean,
    unconditional_mean_cov,
    run_binned_kprop_k2,
    SPIKE_COORD,
)
from .adapter import (
    run_binned_kprop,
    default_binned_kprop_config,
    config_summary,
    extract_mean,
)

__all__ = [
    # K=2 core
    "BinnedK2State", "normal_interval_stats", "find_bin", "safe_bin_representative",
    "symmetrize", "project_to_psd", "make_gaussian_edges", "make_relu_post_edges",
    "lloyd_max_edges", "gaussian_initial_state", "linear_step_k2", "relu_step_k2",
    "unconditional_mean", "unconditional_mean_cov", "run_binned_kprop_k2", "SPIKE_COORD",
    # MLP adapter
    "run_binned_kprop", "default_binned_kprop_config", "config_summary", "extract_mean",
]

# General K > 2 hooks into ordinary harmonic kprop live in ``.kprop_hook`` (torch):
#     from Mecha_preds.binned_kprop.kprop_hook import (
#         bulk_relu_kprop, bulk_relu_kprop_tower, BinnedKState, relu_step_k_general)

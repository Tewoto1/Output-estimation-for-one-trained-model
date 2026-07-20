"""simple_spikeprop -- SIMPLE SPIKE-PROP: constant bulk K=2 + exact 1-d spike recursion.

The minimal mechanistic predictor for the coordinate-spike class ``M = W + e_1 e_1^T``
(same models as ``..binned_kprop`` / ``..analytic_kprop``). Two ingredients only:

  1. the spike coordinate is tracked as a full nonparametric 1-d law on a grid,
     driven by the exact scalar channel recursion
     ``p' = phi_omega * (c ReLU + mu)_# p`` with ``mu = w^T m``, ``omega^2 = w^T Sigma w``
     read off the tracked bulk moments (the affine_knee recursion, per network);
  2. the bulk is ONE unconditional Gaussian ``N(m, Sigma)`` -- mean and covariance
     "held constant" in the spike value (no bins, no affine family, no conditioning),
     pushed by the exact rank-2 Gaussian-ReLU map, with the linear layer's variance
     swap re-aggregated on both sides (``omega^2`` bulk->spike, ``Var(S) u u^T``
     spike->bulk).

By the CONST error accounting both dropped structures (cross-covariance / conditional
dependence) enter propagated means at O(1/n) per coordinate, so the predicted output
mean should follow the same MSE ~ n^{-2} width law as binned/analytic, with a larger
constant. This package is the deliberate ablation floor for that pair:
CONST (here) < AFFINE (analytic) < BINNED (many bins).

    from Mecha_preds.simple_spikeprop import run_simple_spikeprop
    pred = run_simple_spikeprop(model)["mean"]

The core (``core.py``) is numpy/scipy and torch-free; only the ``adapter`` touches
torch. Selftest: ``python -m Mecha_preds.simple_spikeprop.selftest``.
"""
from .core import (
    SpikeLaw,
    SPIKE_COORD,
    gaussian_spike_law,
    channel_push,
    relu_law,
    law_mass,
    law_moments,
    unconditional_mean,
    unconditional_mean_cov,
    run_simple_spikeprop_core,
)
from .adapter import (
    run_simple_spikeprop,
    default_simple_spikeprop_config,
    config_summary,
    extract_mean,
)

__all__ = [
    # core
    "SpikeLaw", "SPIKE_COORD",
    "gaussian_spike_law", "channel_push", "relu_law", "law_mass", "law_moments",
    "unconditional_mean", "unconditional_mean_cov", "run_simple_spikeprop_core",
    # MLP adapter
    "run_simple_spikeprop", "default_simple_spikeprop_config", "config_summary",
    "extract_mean",
]

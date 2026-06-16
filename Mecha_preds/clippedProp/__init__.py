"""clippedProp -- structured propagation with a clamped-Gaussian all-ones channel.

A fourth mechanistic predictor (alongside ``cumulants`` / ``cumulants.skprop`` /
``cumulants.shkprop``), specialised to mean-shifted ReLU MLPs. It propagates the
structured law

    X = s * u + z ,   u = 1_d / sqrt(d) ,   s = max(lo, N(m, v)) ,   z ~ N(mu_z, Sigma_z)

where the all-ones component ``s`` is tracked EXPLICITLY as a clamped ("rectified")
Gaussian and the perpendicular part ``z`` as a (cross-correlated) Gaussian. A
coherent O(1) mean shift -- the failure mode of vanilla k=2 cumulant propagation --
lives entirely in the scalar channel ``s``, which the ReLU step conditions on
(Gauss-Hermite over ``g``) so that ``z | g`` is incoherent again and the standard
Gaussian-ReLU covariance is accurate per node.

    from model import MLP
    from Mecha_preds.clippedProp import run_clipped
    pred = run_clipped(model, config={"n_nodes": 21, "relu_cov": "exact"})["mean"]

Layers: ``linear_layer`` / ``mean_subtraction_layer`` / ``relu_layer`` (in
``layers``); the state + split/refit closure is ``ClippedState`` (in ``state``);
the clamped-Gaussian moment maps are in ``scalar``.
"""
from .scalar import rect_gauss_moments, fit_rect_gauss, clipped_cross_beta
from .state import ClippedState
from .layers import (
    linear_layer,
    linear_output_moments,
    mean_subtraction_layer,
    relu_layer,
)
from .propagate import clipped_mlp_forward
from .adapter import run_clipped, default_clipped_config, clipped_config_summary

__all__ = [
    "rect_gauss_moments", "fit_rect_gauss", "clipped_cross_beta",
    "ClippedState",
    "linear_layer", "linear_output_moments", "mean_subtraction_layer", "relu_layer",
    "clipped_mlp_forward",
    "run_clipped", "default_clipped_config", "clipped_config_summary",
]

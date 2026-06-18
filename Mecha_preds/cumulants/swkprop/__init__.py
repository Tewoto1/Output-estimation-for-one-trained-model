"""swkprop -- Shifted-Weight K-Propagation (SW-KPROP) as a mechanistic predictor.

Predicts ``E[model(X)]`` over ``X ~ N(0, I)`` for ReLU MLPs whose hidden matrices are
mean-shifted, ``M = W - (1/sqrt(n)) 11^T``. Vanilla kprop truncates by total cumulant
order, but the shift amplifies the all-ones ("special") direction by ~sqrt(n) each
layer, so its high-order cumulants are NOT small and must be kept. SW-KPROP works in the
split (special / transverse) basis, keeps the special direction to output rank ``R``
(2 = exact rank-2 Gaussian-ReLU; 3,4 add the amplified cumulants d3,d4 via the
Gram-Charlier closure), and keeps the transverse covariance dense and exact.

    from Mecha_preds.cumulants.swkprop import run_sw_kprop
    pred = run_sw_kprop(model, config={"R": 3}, device="cuda")["mean"]

``core`` is numpy/scipy only (torch-free) so it imports and tests without torch; the
``adapter`` wraps a torch ``model.MLP`` and can route the dense congruence to CUDA.
"""
from .core import (
    State,
    initial_state,
    linear_step,
    relu_step,
    special_mode_quadrature,
    sw_kprop_predict,
)
from .relu import relu_moments_1d, exact_relu_covariance
from .adapter import (
    run_sw_kprop,
    default_sw_kprop_config,
    config_summary,
    extract_mean,
)

__all__ = [
    "run_sw_kprop", "default_sw_kprop_config", "config_summary", "extract_mean",
    "sw_kprop_predict", "State", "initial_state", "linear_step", "relu_step",
    "special_mode_quadrature", "relu_moments_1d", "exact_relu_covariance",
]

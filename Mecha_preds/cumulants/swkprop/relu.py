"""relu.py -- backward-compatibility shim.

The Gaussian-ReLU integrals used to live here; they now live in the canonical, neutrally
named ``Mecha_preds.cumulants.relu_integrals`` (so the shared kernel does not sit inside one
predictor's sub-package). This module just re-exports them, so existing imports such as
``from ..swkprop.relu import relu_moments_1d, exact_relu_covariance, _phi, _Phi`` keep working.

Prefer importing from ``..relu_integrals`` directly in new code.
"""
from ..relu_integrals import (  # noqa: F401  (re-export)
    _phi, _Phi, _bvn_pdf, bvn_cdf, relu_moments_1d,
    _relu_cross_moment_perfect_corr, exact_relu_covariance,
    _DEFAULT_VAR_EPS, _NEG_RTOL, _RHO_TOL, _RHO_VALID_TOL,
)

__all__ = [
    "_phi", "_Phi", "_bvn_pdf", "bvn_cdf", "relu_moments_1d",
    "_relu_cross_moment_perfect_corr", "exact_relu_covariance",
]

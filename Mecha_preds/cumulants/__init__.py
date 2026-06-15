"""cumulants -- cumulant propagation ("kprop") as a mechanistic predictor.

Hand ``run_cumulants`` a trained ``model.MLP`` and it predicts the output mean over
``X ~ N(0, I)`` with the real algorithm vendored in ``.kprop`` (no sampling). It
supports both the normal harmonic path (any ``k_max``) and the exact bivariate
ReLU covariance at ``k_max==2`` (``config={"k_max": 2, "exact_relu_cov": True}``).

    from model import MLP
    from Mecha_preds.cumulants import run_cumulants
    model, _ = MLP.load("checkpoints/zero_d3_w128_seed0_final.pt")
    pred = run_cumulants(model)["mean"]            # default k_max=3
    pred_exact = run_cumulants(model, config={"k_max": 2, "exact_relu_cov": True})["mean"]

``metrics`` provides the Monte-Carlo reference (`estimate_empirical_mean`) and the
comparison (`compare_means`); `run_comparison` is the width-sweep CLI.
"""
# The vendored kprop targets Python >= 3.12. On older interpreters, activate the
# source-rewrite import shim owned by skprop BEFORE importing the adapter (which
# imports kprop). This keeps kprop/ pristine; on >=3.12 the block is skipped and
# skprop is not eagerly imported, so production behaviour is unchanged.
import sys as _sys
if _sys.version_info < (3, 12):
    from .skprop._compat import install as _install_kprop_compat
    _install_kprop_compat()
del _sys

from .adapter import (
    run_cumulants,
    model_to_kprop,
    default_cumulant_config,
    config_summary,
    extract_mean,
)
from .metrics import estimate_empirical_mean, compare_means

__all__ = [
    "run_cumulants", "model_to_kprop", "default_cumulant_config", "config_summary",
    "extract_mean", "estimate_empirical_mean", "compare_means",
]

# Structured power KPROP (spike-aware: latent conditioning + power-cumulant
# residual) lives in ``.skprop`` -- import from there to keep this namespace
# stable:  from Mecha_preds.cumulants.skprop import run_structured_cumulants

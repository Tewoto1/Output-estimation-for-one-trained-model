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
# source-rewrite import shim BEFORE importing the adapter (which imports kprop).
# This keeps kprop/ pristine; on >=3.12 the block is skipped, so production
# behaviour is unchanged.
import sys as _sys
if _sys.version_info < (3, 12):
    from ._kprop_compat import install as _install_kprop_compat
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
from .exact_meanprop import run_exact_meanprop, relu_gaussian_moments

__all__ = [
    "run_cumulants", "model_to_kprop", "default_cumulant_config", "config_summary",
    "extract_mean", "estimate_empirical_mean", "compare_means",
    "run_exact_meanprop", "relu_gaussian_moments",
]

# Shifted-weight KPROP (exact rank-2 plus Edgeworth special cumulants for the all-ones /
# -1/sqrt(n) weight shift) lives in ``.swkprop``:
#     from Mecha_preds.cumulants.swkprop import run_sw_kprop
# SPIKE-KPROP generalizes it to an ARBITRARY unit spike direction v (e.g. localized e1 or
# flat 1/sqrt(n) 1) for the O(1)-eigenvalue spike M = W' + theta v v^T; lives in ``.spikekprop``:
#     from Mecha_preds.cumulants.spikekprop import run_spike_kprop

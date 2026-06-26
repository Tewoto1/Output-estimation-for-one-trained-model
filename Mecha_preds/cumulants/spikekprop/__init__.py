"""spikekprop -- Spike K-Propagation: direction-general split-basis cumulant propagation.

Generalizes SW-KPROP (``..swkprop``) from the hardwired all-ones direction to an ARBITRARY
unit spike direction ``v`` and the *O(1)-eigenvalue* spike regime ``M = W' + theta v v^T``
(``|theta| <= n^{o(1)}``). It retains the spike-direction cumulants ``C(v,...,v) = kappa_p(S)``
of the special mode ``S = v . X`` up to order ``R`` -- exactly the trace-projection terms that
survive for a LOCALIZED spike (``v = e1``: ``sum_i |v_i|^r = 1``) and vanish for a FLAT spike
(``v = 1/sqrt(n) 1``: directional cumulants ``= O(n^{2-r})``).

    from model import MLP
    from Mecha_preds.cumulants.spikekprop import run_spike_kprop
    model, _ = MLP.load("checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt")
    pred  = run_spike_kprop(model, spike_dir="e1")["mean"]                  # R=2 (exact rank-2)
    pred4 = run_spike_kprop(model, spike_dir="e1", config={"R": 4})["mean"] # + d3,d4 along v

``spike_dir`` is ``"e1"``, ``"ones"``, or a vector. Setting ``spike_dir="ones"`` reproduces
SW-KPROP. ReLU numerics reuse the canonical ``..relu_integrals`` (validated). Needs Python >= 3.12 or the
kprop-compat shim only transitively (the spikekprop core itself is torch-free numpy/scipy).
"""
from .core import (
    spike_kprop_predict, unit_vector,
    relu_step_edgeworth, bivariate_relu_wick,   # analytic Gauss-Hermite-free ReLU step
)
from .adapter import (
    run_spike_kprop,
    default_spike_kprop_config,
    config_summary,
    extract_mean,
)

__all__ = [
    "run_spike_kprop", "spike_kprop_predict", "unit_vector",
    "relu_step_edgeworth", "bivariate_relu_wick",
    "default_spike_kprop_config", "config_summary", "extract_mean",
]

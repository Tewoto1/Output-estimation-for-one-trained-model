"""analysis -- mechanistic-interpretability / circuit tools for the study MLP.

Everything lives in `analysis/Tools/` and is re-exported here, so
`from analysis import weight_spectrum` and `from analysis.Tools import ...`
both work. Each tool runs individually and returns a dict (no master report).

    common.collect_activations / run_with_intervention   (primitives)
    pca.activation_pca / pca.weight_spectrum             (PCA, weight SVD)
    ablation.ablate / ablation.neuron_importance         (causal knockout)
    patching.activation_patch                            (activation patching)
    attribution.direct_contributions / output_lens       (direct attribution / logit-lens)
    weight_structure.*                                   (Q2: rank-1 -mu spike metrics)
"""
from .Tools import (
    collect_activations, run_with_intervention,
    activation_pca, weight_spectrum,
    ablate, neuron_importance,
    activation_patch,
    direct_contributions, output_lens, cumulants_sv,
)
from .Tools.weight_structure import (
    mean_prev_post, weight_structure_metrics, W_last, W_first,
    layer_stats, grad_flow,
)
from .Tools.cumulants_sv import (
    reduce_sv, stream_collective_coordinates, AB_REPORT,
    predicted_slope_literal, predicted_slope_heuristic, fit_loglog_slope,
)

__all__ = [
    "collect_activations", "run_with_intervention",
    "activation_pca", "weight_spectrum",
    "ablate", "neuron_importance",
    "activation_patch",
    "direct_contributions", "output_lens",
    "mean_prev_post", "weight_structure_metrics", "W_last", "W_first",
    "layer_stats", "grad_flow", "cumulants_sv",
    "reduce_sv", "stream_collective_coordinates", "AB_REPORT",
    "predicted_slope_literal", "predicted_slope_heuristic", "fit_loglog_slope",
]

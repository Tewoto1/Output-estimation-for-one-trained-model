"""analysis.Tools -- general, task-agnostic circuit tools.

Each tool works on any `model.MLP` + any input batch, runs individually, and
returns a dict (no master "report" step):

    common.collect_activations / run_with_intervention   (primitives)
    pca.activation_pca / pca.weight_spectrum             (PCA, weight SVD)
    ablation.ablate / ablation.neuron_importance         (causal knockout)
    patching.activation_patch                            (activation patching)
    attribution.direct_contributions / output_lens       (direct attribution / logit-lens)
    weight_structure.*                                   (Q2 rank-1 -mu spike metrics)

Example:
    from model import MLP
    from analysis import weight_spectrum, neuron_importance, activation_pca
    import torch
    m, _ = MLP.load("checkpoints/noiseless_Layerless/readout-trainable_d2_w64_seed0_final.pt")
    print(weight_spectrum(m)["readout"]["effective_rank"])
    x = torch.randn(4096, m.cfg.input_dim)
    print(activation_pca(m, x, layer=-1)["top_explained"])
"""
from .common import collect_activations, run_with_intervention
from .pca import activation_pca, weight_spectrum
from .ablation import ablate, neuron_importance
from .patching import activation_patch
from .attribution import direct_contributions, output_lens
import cumulants_sv

__all__ = [
    "collect_activations", "run_with_intervention",
    "activation_pca", "weight_spectrum",
    "ablate", "neuron_importance",
    "activation_patch",
    "direct_contributions", "output_lens",
    "cumulants_sv"
]

"""shkprop -- symbolic hidden-mode KPROP (scalar h, k_max = 2).

A third cumulant-propagation predictor alongside vanilla ``kprop`` and the
spike-aware ``skprop``. It carries the hidden mode ``h`` SYMBOLICALLY through the
network as truncated polynomial jets in ``dh`` (instead of re-running kprop at
quadrature nodes of ``h`` and averaging, which is what ``skprop`` does). Linear
layers are exact in cumulant space; the ReLU layer is the only approximation
(conditional-Gaussian residual closure); the hidden mode is marginalized from its
cumulants at the end. See ``symbolic_hidden_mode_kprop_implementation_spec``.

    from Mecha_preds.cumulants.shkprop import run_symbolic_cumulants
    pred  = run_symbolic_cumulants(model, config={"latent": "ones"})["mean"]   # h = 1^T X / sqrt(n)
    pred0 = run_symbolic_cumulants(model, config={"latent": "none"})["mean"]   # q=0 -> ordinary k=2 kprop

Scope (the smallest correct core of the spec): scalar hidden mode (q = 1),
``k_max = 2`` (mean + covariance), ReLU activation. Vector h, Edgeworth, dynamic
hidden-mode splitting and projected K3/K4 are out of scope; the method REPORTS
where it approximates (``metadata["approximations"]``) and whether the hidden
degree resolved the tail (``metadata["unresolved_layers"]``).

This package depends only on torch + numpy (NOT on the vendored ``kprop``), so it
imports on any Python and runs on GPU. ``reference`` is a pure-numpy oracle that
mirrors the torch path for cross-checking (the notebook asserts agreement).
"""
from .polyjet import PolyJet, jet_outer, fit_jet
from .hidden import HiddenCumulants
from .relu_moments import relu_m1, relu_m2_diag, relu_pair_matrix
from .symbolic import (
    SymbolicConfig,
    SymbolicState,
    make_input_state,
    linear_pushforward_symbolic,
    activation_relu_symbolic,
    marginalize_hidden_modes,
    tail_diagnostics,
    symbolic_hidden_mode_kprop,
)
from .adapter import (
    run_symbolic_cumulants,
    default_symbolic_config,
    symbolic_config_summary,
)
from . import _reference as reference

__all__ = [
    "PolyJet", "jet_outer", "fit_jet",
    "HiddenCumulants",
    "relu_m1", "relu_m2_diag", "relu_pair_matrix",
    "SymbolicConfig", "SymbolicState", "make_input_state",
    "linear_pushforward_symbolic", "activation_relu_symbolic",
    "marginalize_hidden_modes", "tail_diagnostics", "symbolic_hidden_mode_kprop",
    "run_symbolic_cumulants", "default_symbolic_config", "symbolic_config_summary",
    "reference",
]

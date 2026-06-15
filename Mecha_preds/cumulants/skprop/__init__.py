"""skprop -- structured power KPROP (spike-aware cumulant propagation).

Vanilla kprop fails on meaned/spiked matrices because a low-rank weight
component turns a shared latent into coherent O(1) off-diagonal cumulants;
power cumulants alone only fix repeated-index (diagonal) terms. This package
implements the fix: track the spike-selected latent explicitly (Gauss--Hermite
conditioning) and use the vendored power-cumulant kprop for the residual noise
around it. Error budget: O(n^{-k_max/2}) + quadrature + E_condCLT.

    spikes      -- MP-edge detection of the low-rank structured part of W
    structured  -- structured_mlp_kprop: the algorithm, on a kprop MLP
    adapter     -- run_structured_cumulants: drop-in for run_cumulants on a study MLP
    toy         -- exact closed-form toy model (P_i = a_i H + sigma G_i, phi = z^2,
                   cubic readout): Algorithms A/B/C/D with O(1) / O(1/n) /
                   O(1/n^2) / exact error hierarchy
"""
# Make the vendored kprop importable on Python < 3.12 WITHOUT modifying it (it
# uses PEP 695 `type` aliases / `typing.Self`). This shim source-rewrites kprop at
# import time and is a no-op on >=3.12. It lives here, in skprop -- the predictor
# that depends on kprop -- so kprop/ stays byte-for-byte pristine. Must run before
# the submodule imports below (.structured / .adapter pull in ..kprop).
from . import _compat as _compat  # noqa: F401
_compat.install()

from .spikes import SpikeInfo, detect_spikes, detect_spikes_all_layers, orthonormalize
from .structured import (
    structured_mlp_kprop,
    gauss_hermite_grid,
    condition_gaussian_on_subspace,
)
from .adapter import (
    run_structured_cumulants,
    default_structured_config,
    structured_config_summary,
)
from .toy import ToyModel, make_toy, error_sweep

__all__ = [
    "SpikeInfo", "detect_spikes", "detect_spikes_all_layers", "orthonormalize",
    "structured_mlp_kprop", "gauss_hermite_grid", "condition_gaussian_on_subspace",
    "run_structured_cumulants", "default_structured_config", "structured_config_summary",
    "ToyModel", "make_toy", "error_sweep",
]

"""_relu.py -- numpy/scipy Gaussian-ReLU backend for the binned predictor.

The binned core needs four torch-free helpers that already live (validated) in
``Mecha_preds.cumulants.swkprop.relu``:

    _phi, _Phi               -- standard-normal pdf/cdf (scipy ndtr)
    relu_moments_1d          -- exact univariate ReLU moments
    exact_relu_covariance    -- exact bivariate-Gaussian ReLU covariance (rank-2)

Importing them the normal way (``from ..cumulants.swkprop.relu import ...``) is the
production path and is what runs in the repo env. But that import first executes
``Mecha_preds/cumulants/__init__.py``, which eagerly imports the *torch*-based
``adapter``; on an interpreter without torch installed that raises before reaching
the (torch-FREE) ``swkprop.relu`` module. So we fall back to loading that single
file directly by path -- it has no torch dependency of its own -- which lets the
binned K=2 core import and run with only numpy + scipy (e.g. for the selftest /
scaling sweeps). This mirrors the spirit of the repo's existing ``_kprop_compat``
shim: keep the vendored modules pristine, adapt only at the import boundary.
"""
from __future__ import annotations

try:  # production path (repo env has torch): normal package import
    from ..cumulants.swkprop.relu import (  # type: ignore
        _phi, _Phi, exact_relu_covariance, relu_moments_1d,
    )
except ModuleNotFoundError:  # torch absent -> load the torch-free module by path
    import importlib.util as _ilu
    import pathlib as _pl

    _relu_path = (
        _pl.Path(__file__).resolve().parent.parent
        / "cumulants" / "swkprop" / "relu.py"
    )
    _spec = _ilu.spec_from_file_location("_binned_kprop_swkprop_relu", _relu_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not load swkprop ReLU backend from {_relu_path}")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _phi = _mod._phi
    _Phi = _mod._Phi
    exact_relu_covariance = _mod.exact_relu_covariance
    relu_moments_1d = _mod.relu_moments_1d

__all__ = ["_phi", "_Phi", "exact_relu_covariance", "relu_moments_1d"]

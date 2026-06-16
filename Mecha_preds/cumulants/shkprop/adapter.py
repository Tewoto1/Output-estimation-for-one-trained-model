"""adapter.py -- run SYMBOLIC hidden-mode cumulant propagation on a study ``model.MLP``.

Drop-in counterpart of ``Mecha_preds.cumulants.run_cumulants`` /
``skprop.run_structured_cumulants``: same input distribution ``X ~ N(0, I)``, same
float64 policy, same ``{"mean", "metadata", ...}`` return shape -- but the prediction
comes from ``symbolic_hidden_mode_kprop`` (the hidden mode carried symbolically as
polynomial jets) instead of one vanilla ``mlp_kprop`` call or per-node averaging.

    from Mecha_preds.cumulants.shkprop import run_symbolic_cumulants
    pred = run_symbolic_cumulants(model, config={"latent": "ones"})["mean"]

``latent`` selects the scalar hidden mode ``h = V^T X``:
    "ones"  (default)  V = all-ones / sqrt(d)   -- the planted latent of the
                       shifted-mean kprop study (h = 1^T X / sqrt(n)).
    "none"             q = 0: reduces to ordinary k=2 ReLU kprop.
    explicit array     pass ``config={"direction": V}`` for any unit direction.

Unlike skprop (which averages vanilla kprop over Gauss-Hermite nodes of h), this
keeps the h-dependence as a jet and marginalizes once at the end -- so a single
run reproduces the node-average to the hidden-degree truncation order.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from model import MLP as StudyMLP
from .symbolic import SymbolicConfig, symbolic_hidden_mode_kprop

logger = logging.getLogger("shkprop_adapter")

# Keys forwarded to SymbolicConfig (device/dtype are passed explicitly, not here).
_CFG_KEYS = {
    "k_max", "hidden_degree_initial", "hidden_degree_max", "hidden_tail_tol",
    "hidden_tail_band", "auto_refine", "full_covariance", "activation_method",
    "n_gl", "collocation_span", "collocation_oversample", "n_quad_margin",
    "var_floor",
}


def default_symbolic_config() -> dict:
    """Default symbolic-kprop config (spec defaults + adapter knobs)."""
    return {
        "k_max": 2,
        "hidden_degree_initial": 6,
        "hidden_degree_max": 12,
        "hidden_tail_tol": 1e-4,
        "hidden_tail_band": 2,
        "auto_refine": True,
        "full_covariance": True,
        "activation_method": "relu_gaussian_exact",
        "n_gl": 64,
        "collocation_span": 4.0,
        "collocation_oversample": 3,
        "n_quad_margin": 6,
        "var_floor": 1e-12,
        "latent": "ones",        # "ones" | "none" | (use "direction" for explicit V)
        "direction": None,       # explicit unit latent direction (overrides "latent")
    }


def symbolic_config_summary(cfg: dict) -> str:
    lat = "explicit" if cfg.get("direction") is not None else cfg.get("latent")
    return (f"k_max={cfg['k_max']},p0={cfg['hidden_degree_initial']},"
            f"pmax={cfg['hidden_degree_max']},tail_tol={cfg['hidden_tail_tol']},"
            f"full_cov={cfg['full_covariance']},latent={lat},n_gl={cfg['n_gl']}")


def _layers_from_model(model: StudyMLP, device, dtype):
    """('linear', W, b) for each hidden Linear, a ('relu',) after each, then readout."""
    layers = []
    for lin in model.hidden_layers:
        W = lin.weight.detach().to(device=device, dtype=dtype)
        b = lin.bias.detach().to(device=device, dtype=dtype) if lin.bias is not None else None
        layers.append(("linear", W, b))
        layers.append(("relu",))
    ro = model.readout
    Wr = ro.weight.detach().to(device=device, dtype=dtype)
    br = ro.bias.detach().to(device=device, dtype=dtype) if ro.bias is not None else None
    layers.append(("linear", Wr, br))
    return layers


def _direction(cfg: dict, input_dim: int, device, dtype) -> Optional[torch.Tensor]:
    if cfg.get("direction") is not None:
        return torch.as_tensor(cfg["direction"], device=device, dtype=dtype).reshape(-1)
    latent = cfg.get("latent", "ones")
    if latent in (None, "none", "q0"):
        return None
    if latent == "ones":
        return torch.ones(input_dim, device=device, dtype=dtype) / np.sqrt(input_dim)
    raise ValueError(f"unknown latent {latent!r}; use 'ones', 'none', or pass 'direction'")


@torch.no_grad()
def run_symbolic_cumulants(model: StudyMLP, input_dim: Optional[int] = None,
                           config: Optional[dict] = None, *, device: str = "cpu",
                           debug: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, I)`` via symbolic hidden-mode kprop."""
    if not isinstance(model, StudyMLP):
        raise TypeError(f"run_symbolic_cumulants expects a model.MLP, got {type(model)!r}")
    if model.cfg.activation != "relu":
        raise ValueError("symbolic hidden-mode kprop (this scope) supports ReLU only; "
                         f"got activation={model.cfg.activation!r}")
    cfg = default_symbolic_config()
    if config:
        cfg.update(config)
    if input_dim is None:
        input_dim = model.cfg.input_dim

    dtype = cfg["dtype"] if isinstance(cfg["dtype"], torch.dtype) else torch.float64
    sym_cfg = SymbolicConfig(**{k: cfg[k] for k in _CFG_KEYS if k in cfg},
                             device=device, dtype=dtype)
    layers = _layers_from_model(model, device, dtype)
    direction = _direction(cfg, input_dim, device, dtype)

    res = symbolic_hidden_mode_kprop(layers, input_dim, sym_cfg, direction=direction)
    mean = res["mean"].detach().cpu().double().numpy().reshape(-1)
    if debug:
        logger.info("symbolic mean shape=%s p=%d unresolved=%s",
                    mean.shape, res["hidden_degree"], res["unresolved_layers"])

    return {
        "raw_output": res,
        "mean": mean,
        "metadata": {
            "config": symbolic_config_summary(cfg),
            "config_dict": cfg,
            "hidden_degree": int(res["hidden_degree"]),
            "q": 0 if direction is None else 1,
            "unresolved_layers": res["unresolved_layers"],
            "tail_scores": [d["tail_score"] for d in res["layer_diagnostics"]],
            "approximations": res["approximations"],
            "input_dim": int(input_dim),
            "output_dim": int(mean.shape[0]),
        },
    }

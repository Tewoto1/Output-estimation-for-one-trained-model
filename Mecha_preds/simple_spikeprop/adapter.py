"""adapter.py -- run SIMPLE SPIKE-PROP on a study ``model.MLP``.

Drop-in like ``run_binned_kprop`` / ``run_analytic_kprop``:

    from model import MLP
    from Mecha_preds.simple_spikeprop import run_simple_spikeprop
    model, _ = MLP.load("checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt")
    pred  = run_simple_spikeprop(model)["mean"]
    pred2 = run_simple_spikeprop(model, config={"num_grid": 4001})["mean"]

There is NO structural hyperparameter (that is the point): the bulk is one
unconditional Gaussian, the spike one grid law. ``num_grid`` / ``span`` only set the
1-d quadrature resolution and converge fast. The coordinate spike acts on ``e_1``
(hidden coordinate 0); pass ``add_spike=True`` to ADD ``spike_theta * e_c e_c^T`` to
each (square) hidden matrix when the stored weights do not already contain it
(default ``False`` -- assume, like binned/spikekprop, the spike is baked in).
Requires square hidden layers (``input_dim == hidden_dim``).

The core is numpy/scipy (torch-free); only the weight extraction here touches torch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import run_simple_spikeprop_core, SPIKE_COORD


def default_simple_spikeprop_config() -> dict:
    """Simple spike-prop knobs -- quadrature only, no structural hyperparameter.
    ``num_grid`` = nodes of the 1-d spike grid per layer; ``span`` = Gaussian tail
    span in sigmas beyond the mixture-component range; ``input_std`` = Gaussian
    input scale."""
    return {"num_grid": 2001, "span": 8.0, "input_std": 1.0}


def config_summary(config: Optional[dict] = None) -> str:
    c = default_simple_spikeprop_config()
    if config:
        c.update(config)
    return (f"SIMPLE-SPIKEPROP(const bulk K=2, num_grid={c['num_grid']}, "
            f"span={c['span']}, input_std={c['input_std']})")


def _weights_from_model(model):
    """[(W, b), ...] float64 numpy in forward order: hidden_0..hidden_{d-1}, readout.
    Mirrors the binned_kprop/spikekprop extraction (W shape (out, in); b or None)."""
    layers = list(model.hidden_layers) + [model.readout]
    out = []
    for lin in layers:
        W = lin.weight.detach().cpu().double().numpy()
        b = None if lin.bias is None else lin.bias.detach().cpu().double().numpy()
        out.append((W, b))
    return out


def run_simple_spikeprop(model, input_dim: Optional[int] = None,
                         config: Optional[dict] = None, *,
                         add_spike: bool = False, spike_coord: int = SPIKE_COORD,
                         spike_theta: float = 1.0, collect: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, I)`` by simple spike-prop (constant
    bulk Gaussian + exact 1-d spike-channel recursion).

    ``config`` keys: ``num_grid``, ``span``, ``input_std`` (quadrature knobs only).
    ``add_spike=True`` adds ``spike_theta * e_{spike_coord} e^T`` to each square
    hidden matrix (use when the spike is not already in the stored weights).
    Returns ``{"mean", "cov", "metadata", ...}``.
    """
    cfg = default_simple_spikeprop_config()
    if config:
        cfg.update(config)
    if model.cfg.activation != "relu":
        raise ValueError(f"simple spike-prop supports ReLU only; got {model.cfg.activation!r}")
    if input_dim is None:
        input_dim = model.cfg.input_dim

    weights = _weights_from_model(model)
    n_hidden = len(weights) - 1
    for li in range(n_hidden):
        W, _b = weights[li]
        if W.shape != (input_dim, input_dim):
            raise ValueError(
                f"simple spike-prop needs square hidden layers ({input_dim},{input_dim}); "
                f"hidden layer {li} has shape {W.shape}. (Train-to-zero models have "
                f"input_dim == hidden_dim; pass input_dim explicitly if needed.)")
        if add_spike:
            W = W.copy()
            W[spike_coord, spike_coord] += spike_theta
            weights[li] = (W, _b)

    res = run_simple_spikeprop_core(
        weights, input_dim,
        num_grid=int(cfg["num_grid"]), span=float(cfg["span"]),
        input_std=float(cfg["input_std"]), collect=collect)
    res["metadata"]["config"] = config_summary(config)
    res["metadata"]["add_spike"] = bool(add_spike)
    if add_spike:
        res["metadata"]["spike_coord"] = int(spike_coord)
        res["metadata"]["spike_theta"] = float(spike_theta)
    return res


def extract_mean(result: dict) -> np.ndarray:
    return np.asarray(result["mean"], dtype=np.float64).reshape(-1)

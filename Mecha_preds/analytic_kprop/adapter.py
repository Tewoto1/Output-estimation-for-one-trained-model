"""adapter.py -- run ANALYTIC AFFINE K=2 propagation on a study ``model.MLP``.

Drop-in like ``run_binned_kprop`` / ``run_spike_kprop``:

    from model import MLP
    from Mecha_preds.analytic_kprop import run_analytic_kprop
    model, _ = MLP.load("checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt")
    pred  = run_analytic_kprop(model)["mean"]                        # 40 nodes (default)
    pred2 = run_analytic_kprop(model, config={"num_nodes": 80})["mean"]

``num_nodes`` is the adjustable hyperparameter (total signed quadrature cells per
layer). The coordinate spike acts on ``e_1`` (hidden coordinate 0); pass
``add_spike=True`` to ADD ``spike_theta * e_c e_c^T`` to each (square) hidden
matrix when the stored weights do not already contain it. Requires square hidden
layers (``input_dim == hidden_dim``), as the train-to-zero models satisfy.

The core is numpy/scipy (torch-free); only the weight extraction here touches torch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import run_analytic_kprop_k2, SPIKE_COORD


def default_analytic_kprop_config() -> dict:
    """Analytic-kprop knobs. ``num_nodes`` = total signed scalar quadrature cells per
    layer (THE hyperparameter; split across the sign proportionally to mixture mass
    unless ``num_nodes_neg``/``num_nodes_pos`` override); ``grid`` places the cells
    ('w2' = Lloyd-Max on the exact scalar mixture, 'uniform' = equal width);
    ``bulk_relu`` is the per-node Gaussian-ReLU backend ('exact' = exact bivariate
    covariance, 'gain' = leading-order gain, 'kprop' = harmonic kprop, torch);
    ``cov_intercept`` = 'mc' (moment-conservative, paper eq 90) or 'ls';
    ``input_std`` is the Gaussian input scale."""
    return {"num_nodes": 40, "num_nodes_neg": None, "num_nodes_pos": None,
            "grid": "w2", "bulk_relu": "exact", "cov_intercept": "mc",
            "input_std": 1.0, "diagnostics": False}


def config_summary(config: Optional[dict] = None) -> str:
    c = default_analytic_kprop_config()
    if config:
        c.update(config)
    return (f"ANALYTIC-KPROP(K=2, num_nodes={c['num_nodes']}, grid={c['grid']}, "
            f"bulk_relu={c['bulk_relu']}, cov_intercept={c['cov_intercept']})")


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


def run_analytic_kprop(model, input_dim: Optional[int] = None, config: Optional[dict] = None,
                       *, add_spike: bool = False, spike_coord: int = SPIKE_COORD,
                       spike_theta: float = 1.0, collect: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, I)`` by analytic affine K=2 propagation.

    ``config`` keys: see ``default_analytic_kprop_config``. ``add_spike=True`` adds
    ``spike_theta * e_{spike_coord} e^T`` to each square hidden matrix (use when the
    spike is not already in the stored weights). Returns ``{"mean", "metadata", ...}``.
    """
    cfg = default_analytic_kprop_config()
    if config:
        cfg.update(config)
    if model.cfg.activation != "relu":
        raise ValueError(f"analytic kprop supports ReLU only; got {model.cfg.activation!r}")
    if input_dim is None:
        input_dim = model.cfg.input_dim

    weights = _weights_from_model(model)
    n_hidden = len(weights) - 1
    for li in range(n_hidden):
        W, _b = weights[li]
        if W.shape != (input_dim, input_dim):
            raise ValueError(
                f"analytic kprop needs square hidden layers ({input_dim},{input_dim}); hidden "
                f"layer {li} has shape {W.shape}. (Train-to-zero models have input_dim == "
                f"hidden_dim; pass input_dim explicitly if needed.)")
        if add_spike:
            W = W.copy()
            W[spike_coord, spike_coord] += spike_theta
            weights[li] = (W, _b)

    res = run_analytic_kprop_k2(
        weights, input_dim, num_nodes=int(cfg["num_nodes"]),
        num_nodes_neg=(None if cfg.get("num_nodes_neg") is None else int(cfg["num_nodes_neg"])),
        num_nodes_pos=(None if cfg.get("num_nodes_pos") is None else int(cfg["num_nodes_pos"])),
        grid=str(cfg["grid"]), bulk_relu=str(cfg["bulk_relu"]),
        cov_intercept=str(cfg["cov_intercept"]), input_std=float(cfg["input_std"]),
        diagnostics=bool(cfg.get("diagnostics", False)), collect=collect)
    res["metadata"]["config"] = config_summary(config)
    res["metadata"]["add_spike"] = bool(add_spike)
    if add_spike:
        res["metadata"]["spike_coord"] = int(spike_coord)
        res["metadata"]["spike_theta"] = float(spike_theta)
    return res


def extract_mean(result: dict) -> np.ndarray:
    return np.asarray(result["mean"], dtype=np.float64).reshape(-1)

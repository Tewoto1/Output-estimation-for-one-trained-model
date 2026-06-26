"""adapter.py -- run COORDINATE-SPIKE BINNED kprop (K=2) on a study ``model.MLP``.

Drop-in like ``run_cumulants`` / ``run_spike_kprop``:

    from model import MLP
    from Mecha_preds.binned_kprop import run_binned_kprop
    model, _ = MLP.load("checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt")
    pred  = run_binned_kprop(model)["mean"]                       # 21 bins (default)
    pred2 = run_binned_kprop(model, config={"num_bins": 41})["mean"]

``num_bins`` is the adjustable hyperparameter (number of spike bins; THIS IS THE K=2
predictor). The coordinate spike acts on ``e_1`` (hidden coordinate 0); pass
``add_spike=True`` to ADD ``spike_theta * e_c e_c^T`` to each (square) hidden matrix
when the stored weights do not already contain it (default ``False`` -- assume, like
``spikekprop``, that the spike is baked into the trained weights). Requires square
hidden layers (``input_dim == hidden_dim``), which the train-to-zero models satisfy.

The K=2 core is numpy/scipy (torch-free); only the weight extraction here touches torch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import run_binned_kprop_k2, SPIKE_COORD


def default_binned_kprop_config() -> dict:
    """Binned-kprop knobs. ``num_bins`` = number of spike bins (the hyperparameter);
    ``num_bins_post`` sizes the post-ReLU grid (default = ``num_bins``); ``bulk_relu``
    is the per-bin bulk-ReLU backend ('exact' = exact bivariate covariance, 'gain' =
    leading-order spec-8.2 gain, 'kprop' = delegate to harmonic kprop); ``input_std``
    is the Gaussian input scale."""
    return {"num_bins": 21, "num_bins_post": None, "bulk_relu": "exact", "input_std": 1.0}


def config_summary(config: Optional[dict] = None) -> str:
    c = default_binned_kprop_config()
    if config:
        c.update(config)
    npost = c["num_bins_post"] if c["num_bins_post"] is not None else c["num_bins"]
    return f"BINNED-KPROP(K=2, num_bins={c['num_bins']}, post={npost}, bulk_relu={c['bulk_relu']})"


def _weights_from_model(model):
    """[(W, b), ...] float64 numpy in forward order: hidden_0..hidden_{d-1}, readout.
    Mirrors spikekprop/exact_meanprop extraction (W shape (out, in); b or None)."""
    layers = list(model.hidden_layers) + [model.readout]
    out = []
    for lin in layers:
        W = lin.weight.detach().cpu().double().numpy()
        b = None if lin.bias is None else lin.bias.detach().cpu().double().numpy()
        out.append((W, b))
    return out


def run_binned_kprop(model, input_dim: Optional[int] = None, config: Optional[dict] = None,
                     *, add_spike: bool = False, spike_coord: int = SPIKE_COORD,
                     spike_theta: float = 1.0, collect: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, I)`` by coordinate-spike binned kprop (K=2).

    ``config`` keys: ``num_bins`` (the hyperparameter), ``num_bins_post``, ``bulk_relu``,
    ``input_std``. ``add_spike=True`` adds ``spike_theta * e_{spike_coord} e^T`` to each
    square hidden matrix (use when the spike is not already in the stored weights).
    Returns ``{"mean", "metadata", ...}``.
    """
    cfg = default_binned_kprop_config()
    if config:
        cfg.update(config)
    if model.cfg.activation != "relu":
        raise ValueError(f"binned kprop supports ReLU only; got {model.cfg.activation!r}")
    if input_dim is None:
        input_dim = model.cfg.input_dim

    weights = _weights_from_model(model)
    n_hidden = len(weights) - 1
    for li in range(n_hidden):
        W, _b = weights[li]
        if W.shape != (input_dim, input_dim):
            raise ValueError(
                f"binned kprop needs square hidden layers ({input_dim},{input_dim}); hidden "
                f"layer {li} has shape {W.shape}. (Train-to-zero models have input_dim == "
                f"hidden_dim; pass input_dim explicitly if needed.)")
        if add_spike:
            W = W.copy()
            W[spike_coord, spike_coord] += spike_theta
            weights[li] = (W, _b)

    res = run_binned_kprop_k2(
        weights, input_dim, num_bins=int(cfg["num_bins"]),
        num_bins_post=(None if cfg["num_bins_post"] is None else int(cfg["num_bins_post"])),
        input_std=float(cfg["input_std"]), bulk_relu=str(cfg["bulk_relu"]),
        collect=collect)
    res["metadata"]["config"] = config_summary(config)
    res["metadata"]["add_spike"] = bool(add_spike)
    if add_spike:
        res["metadata"]["spike_coord"] = int(spike_coord)
        res["metadata"]["spike_theta"] = float(spike_theta)
    return res


def extract_mean(result: dict) -> np.ndarray:
    return np.asarray(result["mean"], dtype=np.float64).reshape(-1)

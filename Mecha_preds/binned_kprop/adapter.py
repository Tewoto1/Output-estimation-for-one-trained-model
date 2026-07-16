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

from .binning import resolve_workers
from .core import run_binned_kprop_k2, SPIKE_COORD


def default_binned_kprop_config() -> dict:
    """Binned-kprop knobs. ``num_bins`` = the POSITIVE-side spike-bin budget (the
    hyperparameter; the ReLU keeps exactly these bins + the zero atom -- no post-ReLU
    re-binning exists anymore). ``grid="wasserstein"`` (default) rebuilds the
    pre-activation grid per layer, split at 0, with a MASS-ADAPTIVE negative side
    (``num_bins_pre_neg=None``; cap ``max_bins_neg``, default ``8 * num_bins``);
    ``grid="fixed"`` is the legacy static Gaussian-quantile grid. ``bulk_relu``
    is the per-bin bulk-ReLU backend ('exact' = exact bivariate covariance, 'gain' =
    leading-order spec-8.2 gain, 'kprop' = delegate to harmonic kprop); ``input_std``
    is the Gaussian input scale; ``workers`` is the per-bin thread count -- ``"auto"``
    (default) parallelizes per machine (CUDA box -> 8, else min(8, cpu_count); env override
    ``BINNED_KPROP_WORKERS``), pass ``1`` for serial. (``num_bins_post`` in old cached
    configs is tolerated and ignored.)"""
    return {"num_bins": 21,
            "num_bins_pre_pos": None, "num_bins_pre_neg": None, "max_bins_neg": None,
            "bulk_relu": "exact", "input_std": 1.0,
            "grid": "wasserstein", "relu_merge": "post", "workers": "auto"}


def config_summary(config: Optional[dict] = None) -> str:
    c = default_binned_kprop_config()
    if config:
        c.update(config)
    nneg = c.get("num_bins_pre_neg")
    return (f"BINNED-KPROP(K=2, num_bins={c['num_bins']}, "
            f"neg={'adaptive' if nneg is None else nneg}, "
            f"grid={c.get('grid', 'wasserstein')}, bulk_relu={c['bulk_relu']}, "
            f"workers={resolve_workers(c.get('workers', 'auto'))})")


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

    ``config`` keys: ``num_bins`` (the positive-side budget -- THE hyperparameter),
    ``bulk_relu``, ``input_std``, and (``grid="wasserstein"`` only) ``num_bins_pre_pos``
    / ``num_bins_pre_neg`` / ``max_bins_neg`` -- positive override (default = ``num_bins``),
    negative override (default = mass-adaptive: same mass-per-bin as the positive side),
    and the adaptive cap (default ``8 * num_bins``). ReLU keeps every positive bin verbatim
    and merges the negatives into the zero atom; there is no post grid.
    ``add_spike=True`` adds ``spike_theta * e_{spike_coord} e^T`` to each
    square hidden matrix (use when the spike is not already in the stored weights).
    Returns ``{"mean", "metadata", ...}``.
    """
    cfg = default_binned_kprop_config()
    if config:
        cfg.update(config)
    cfg.pop("num_bins_post", None)                     # legacy key in old cached configs
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
        grid=str(cfg.get("grid", "wasserstein")),
        num_bins_pre_pos=(None if cfg.get("num_bins_pre_pos") is None else int(cfg["num_bins_pre_pos"])),
        num_bins_pre_neg=(None if cfg.get("num_bins_pre_neg") is None else int(cfg["num_bins_pre_neg"])),
        max_bins_neg=(None if cfg.get("max_bins_neg") is None else int(cfg["max_bins_neg"])),
        input_std=float(cfg["input_std"]), bulk_relu=str(cfg["bulk_relu"]),
        relu_merge=str(cfg.get("relu_merge", "post")),
        workers=cfg.get("workers", "auto"),
        collect=collect)
    res["metadata"]["config"] = config_summary(config)
    res["metadata"]["add_spike"] = bool(add_spike)
    if add_spike:
        res["metadata"]["spike_coord"] = int(spike_coord)
        res["metadata"]["spike_theta"] = float(spike_theta)
    return res


def extract_mean(result: dict) -> np.ndarray:
    return np.asarray(result["mean"], dtype=np.float64).reshape(-1)

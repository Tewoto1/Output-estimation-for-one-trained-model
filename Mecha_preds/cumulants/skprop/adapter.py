"""adapter.py -- run STRUCTURED cumulant propagation on a study ``model.MLP``.

Drop-in counterpart of ``Mecha_preds.cumulants.run_cumulants``: same input
distribution (X ~ N(0, I)), same float64 policy, same return shape -- but the
prediction comes from ``structured_mlp_kprop`` (spike-aware latent conditioning
+ the vendored power-cumulant kprop on the residual) instead of one vanilla
``mlp_kprop`` call.

    from Mecha_preds.cumulants.skprop import run_structured_cumulants
    pred = run_structured_cumulants(model, config={"k_max": 2, "n_nodes": 15})["mean"]

Extra config keys on top of ``default_cumulant_config``:
    n_nodes (15)      Gauss--Hermite nodes per latent dimension
    q_max (1)         max latents auto-detected on the FIRST weight layer
    margin (1.15)     MP-edge multiplier a singular value must beat to count
    directions (None) explicit (input_dim, q) latent directions; skips detection
    deep (False)      also condition on hidden-layer spike channels (k_max==2 only)
    deep_layers/deep_n_nodes/deep_q_max/deep_margin   deep-mode knobs

With q=0 detected (no spike) and deep=False the prediction is EXACTLY the
vanilla ``run_cumulants`` one -- so this can be used unconditionally.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from model import MLP as StudyMLP
from ..adapter import default_cumulant_config, model_to_kprop, KIND_BY_NAME
from ..kprop import Kind
from .structured import structured_mlp_kprop

logger = logging.getLogger("skprop_adapter")


def default_structured_config() -> dict:
    cfg = default_cumulant_config()
    cfg.update({
        "k_max": 2,            # deep mode needs 2; raise for input-latent-only runs
        "factor": False,
        "n_nodes": 15,
        "q_max": 1,
        "margin": 1.15,
        "directions": None,
        "deep": False,
        "deep_layers": None,
        "deep_directions": None,   # {linear_layer_idx: (out_dim, q) channels}, e.g. from DeltaW
        "deep_n_nodes": 9,
        "deep_q_max": 1,
        "deep_margin": 1.3,
    })
    return cfg


def structured_config_summary(cfg: dict) -> str:
    kind = cfg["kind"] if isinstance(cfg["kind"], str) else cfg["kind"].name.lower()
    base = (f"k_max={cfg['k_max']},kind={kind},nodes={cfg['n_nodes']},q_max={cfg['q_max']},"
            f"margin={cfg['margin']},deep={cfg['deep']},use_pK={cfg['use_pK']},"
            f"exact_relu_cov={cfg.get('exact_relu_cov', False)}")
    if cfg["deep"]:
        base += (f",deep_layers={cfg['deep_layers']},deep_nodes={cfg['deep_n_nodes']},"
                 f"deep_q_max={cfg['deep_q_max']},deep_margin={cfg['deep_margin']}")
    return base


@torch.no_grad()
def run_structured_cumulants(model: StudyMLP, input_dim: Optional[int] = None,
                             config: Optional[dict] = None, *, device: str = "cpu",
                             debug: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, I)`` via structured power KPROP."""
    cfg = default_structured_config()
    if config:
        cfg.update(config)
    if isinstance(cfg["kind"], str):
        cfg["kind"] = KIND_BY_NAME[cfg["kind"].lower()]
    if input_dim is None:
        input_dim = model.cfg.input_dim

    kmlp = model_to_kprop(model, device=device)
    in_f = kmlp.Ws[0].weight.shape[1]
    if in_f != input_dim:
        raise ValueError(f"first layer in_features={in_f} != input_dim={input_dim}")

    K_in = {1: torch.zeros(input_dim, device=device, dtype=torch.float64),
            2: torch.eye(input_dim, device=device, dtype=torch.float64)}

    res = structured_mlp_kprop(
        kmlp, K_in, k_max=cfg["k_max"],
        directions=cfg["directions"], q_max=cfg["q_max"], margin=cfg["margin"],
        n_nodes=cfg["n_nodes"],
        deep=cfg["deep"], deep_layers=cfg["deep_layers"],
        deep_directions=cfg["deep_directions"],
        deep_n_nodes=cfg["deep_n_nodes"], deep_q_max=cfg["deep_q_max"],
        deep_margin=cfg["deep_margin"],
        output_d_max=cfg["output_d_max"],
        kind=cfg["kind"], use_avg_metric=cfg["use_avg_metric"],
        factor=cfg["factor"], use_pK=cfg["use_pK"],
        exact_relu_cov=cfg.get("exact_relu_cov", False),
    )

    mean = res["mean"].detach().cpu().double().numpy().reshape(-1)
    if debug:
        logger.info("structured mean shape=%s q=%d branches=%d",
                    mean.shape, res["q"], res["n_branches"])

    return {
        "raw_output": res,
        "mean": mean,
        "metadata": {
            "config": structured_config_summary(cfg),
            "config_dict": cfg,
            "q": res["q"],
            "n_branches": res["n_branches"],
            "spike": res["spikes"].summary() if res["spikes"] is not None else None,
            "deep_spikes": [s.summary() for s in res["deep_spikes"]],
            "input_dim": int(input_dim),
            "output_dim": int(mean.shape[0]),
        },
    }

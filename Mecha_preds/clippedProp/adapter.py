"""adapter.py -- run clippedProp on a study ``model.MLP`` (drop-in like run_cumulants).

    from Mecha_preds.clippedProp import run_clipped
    pred = run_clipped(model, config={"n_nodes": 21, "relu_cov": "exact"})["mean"]

Same contract as ``Mecha_preds.cumulants.run_cumulants``: input ``X ~ N(0, I)``,
float64, returns ``{"raw_output", "mean" (np.ndarray, out_dim), "metadata"}``. The
prediction is the structured clamped-scalar propagation in ``propagate`` rather
than a cumulant tower.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from .propagate import clipped_mlp_forward

logger = logging.getLogger("clippedProp_adapter")


def default_clipped_config() -> dict:
    """Default clippedProp configuration."""
    return {
        "n_nodes": 21,              # Gauss-Hermite nodes for the per-ReLU scalar integral
        "relu_cov": "exact",        # "exact" (bivariate, scipy) | "gain" (leading-order off-diag)
        "clip_after_linear": False, # pre-activation scalar is plain Gaussian (signed)
        "mean_subtract_after": (),  # hidden-block indices to follow with a mean-subtraction layer
        "want_cov": False,          # also return the full output covariance
    }


def clipped_config_summary(cfg: dict) -> str:
    return (f"n_nodes={cfg['n_nodes']},relu_cov={cfg['relu_cov']},"
            f"clip_after_linear={cfg['clip_after_linear']},"
            f"mean_subtract_after={tuple(cfg['mean_subtract_after'])}")


@torch.no_grad()
def run_clipped(model, input_dim: Optional[int] = None, config: Optional[dict] = None,
                *, device: str = "cpu", debug: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, I_input_dim)`` via clippedProp."""
    cfg = default_clipped_config()
    if config:
        cfg.update(config)
    if input_dim is None:
        input_dim = model.cfg.input_dim

    res = clipped_mlp_forward(
        model, input_dim=input_dim,
        n_nodes=cfg["n_nodes"], relu_cov=cfg["relu_cov"],
        clip_after_linear=cfg["clip_after_linear"],
        mean_subtract_after=cfg["mean_subtract_after"],
        want_cov=cfg["want_cov"], device=device,
    )

    mean = res["mean"].detach().cpu().double().numpy().reshape(-1)
    if debug:
        logger.info("clipped mean shape=%s | final state %s", mean.shape, res["final_state"].summary())

    out = {
        "raw_output": res,
        "mean": mean,
        "metadata": {
            "config": clipped_config_summary(cfg),
            "config_dict": cfg,
            "input_dim": int(input_dim),
            "output_dim": int(mean.shape[0]),
            "final_state": res["final_state"].summary(),
        },
    }
    if cfg["want_cov"] and res["cov"] is not None:
        out["cov"] = res["cov"].detach().cpu().double().numpy()
    return out

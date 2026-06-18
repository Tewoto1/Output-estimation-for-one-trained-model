"""adapter.py -- run SW-KPROP on a study ``model.MLP`` (drop-in like run_cumulants).

    from model import MLP
    from Mecha_preds.cumulants.swkprop import run_sw_kprop
    model, _ = MLP.load("checkpoints/kprop_checkpoints/kprop-zero_d3_w128_tol5_seed3_final.pt")
    pred  = run_sw_kprop(model)["mean"]                      # default R=2 (exact rank-2)
    pred3 = run_sw_kprop(model, config={"R": 3})["mean"]     # + 3rd special cumulant
    pred4 = run_sw_kprop(model, config={"R": 4})["mean"]     # + 4th special cumulant

The numerics run in float64 (the repo accuracy policy). The only dense O(n^3) work is
the transverse covariance congruence ``W Sig_perp W^T`` per layer; pass
``device="cuda"`` to route that congruence to the GPU in float64 (CUDA has float64;
Apple MPS does not, so it stays on the CPU there). The ReLU integrals are scipy/CPU.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import sw_kprop_predict


def default_sw_kprop_config() -> dict:
    """SW-KPROP knobs. ``R`` = output rank / highest special cumulant kept
    (2 = exact rank-2 Gaussian-ReLU; 3,4 add the amplified special cumulants via the
    Gram-Charlier closure). ``n_nodes`` = Gauss-Hermite nodes for the special mode."""
    return {"R": 2, "n_nodes": 9, "input_std": 1.0}


def config_summary(config: Optional[dict] = None) -> str:
    c = default_sw_kprop_config()
    if config:
        c.update(config)
    return f"SW-KPROP(R={c['R']}, n_nodes={c['n_nodes']})"


def _weights_from_model(model):
    """[(W, b), ...] float64 numpy in forward order: hidden_0..hidden_{d-1}, readout.
    Mirrors exact_meanprop's extraction (W shape (out, in); b or None)."""
    layers = list(model.hidden_layers) + [model.readout]
    out = []
    for lin in layers:
        W = lin.weight.detach().cpu().double().numpy()
        b = None if lin.bias is None else lin.bias.detach().cpu().double().numpy()
        out.append((W, b))
    return out


def _make_mm(device: str):
    """Matmul backend for the dense congruence. CUDA -> float64 on GPU; else numpy."""
    if device is None or str(device).startswith("cpu"):
        return lambda A, B: A @ B
    try:
        import torch
    except Exception:
        return lambda A, B: A @ B
    dev = torch.device(device)
    if dev.type != "cuda":          # MPS has no float64; keep accuracy on CPU
        return lambda A, B: A @ B

    def mm(A, B):
        ta = torch.as_tensor(A, device=dev, dtype=torch.float64)
        tb = torch.as_tensor(B, device=dev, dtype=torch.float64)
        return (ta @ tb).cpu().numpy()
    return mm


def run_sw_kprop(model, input_dim: Optional[int] = None, config: Optional[dict] = None,
                 *, device: str = "cpu", collect: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, I)`` by Shifted-Weight K-Propagation.

    ``config`` keys: ``R`` (2/3/4), ``n_nodes``, ``input_std``. ``device="cuda"`` routes
    the transverse congruence to the GPU (float64). Returns ``{"mean", "metadata", ...}``.
    """
    cfg = default_sw_kprop_config()
    if config:
        cfg.update(config)
    if model.cfg.activation != "relu":
        raise ValueError(f"SW-KPROP supports ReLU only; got {model.cfg.activation!r}")
    if input_dim is None:
        input_dim = model.cfg.input_dim

    weights = _weights_from_model(model)
    mm = _make_mm(device)
    res = sw_kprop_predict(weights, input_dim, R=int(cfg["R"]), n_nodes=int(cfg["n_nodes"]),
                           input_std=float(cfg["input_std"]), mm=mm, collect=collect)
    res["metadata"]["device"] = str(device)
    res["metadata"]["config"] = config_summary(config)
    return res


def extract_mean(result: dict) -> np.ndarray:
    return np.asarray(result["mean"], dtype=np.float64).reshape(-1)

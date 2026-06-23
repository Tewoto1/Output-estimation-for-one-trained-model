"""adapter.py -- run SPIKE-KPROP on a study ``model.MLP`` (drop-in like run_cumulants).

    from model import MLP
    from Mecha_preds.cumulants.spikekprop import run_spike_kprop
    model, _ = MLP.load("checkpoints/spike_kprop/spike-e1_d3_w128_seed1_final.pt")
    pred  = run_spike_kprop(model, spike_dir="e1")["mean"]                    # R=2 (exact rank-2)
    pred3 = run_spike_kprop(model, spike_dir="e1", config={"R": 3})["mean"]   # + d3 = C(v,v,v)
    pred4 = run_spike_kprop(model, spike_dir="e1", config={"R": 4})["mean"]   # + d4 = C(v,v,v,v)

``spike_dir`` selects the direction the higher cumulants are tracked along: ``"e1"`` (the
localized single-coordinate spike), ``"ones"`` (the flat 1/sqrt(n) 1 all-ones spike), or any
length-``input_dim`` vector. The spike (``theta v v^T`` added to the hidden weights) is assumed
already present in the model; only the DIRECTION is passed here.

Numerics run in float64 (repo accuracy policy). The only dense O(n^3) work is the transverse
covariance congruence ``W Sig_perp W^T`` per layer; pass ``device="cuda"`` to route it to the
GPU in float64 (CUDA has float64; Apple MPS does not, so it stays on the CPU there). The ReLU
integrals are scipy/CPU and reuse SW-KPROP's validated kernel.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .core import spike_kprop_predict


def default_spike_kprop_config() -> dict:
    """SPIKE-KPROP knobs. ``R`` = highest spike-direction cumulant kept (2 = exact rank-2
    Gaussian-ReLU; 3 adds d3 = C(v,v,v); 4 adds d4 = C(v,v,v,v) via the Gram-Charlier closure).
    ``n_nodes`` = Gauss-Hermite nodes for the special mode S = v . X."""
    return {"R": 2, "n_nodes": 9, "input_std": 1.0}


def config_summary(spike_dir, config: Optional[dict] = None) -> str:
    c = default_spike_kprop_config()
    if config:
        c.update(config)
    d = spike_dir if isinstance(spike_dir, str) else "custom"
    return f"SPIKE-KPROP(dir={d}, R={c['R']}, n_nodes={c['n_nodes']})"


def _weights_from_model(model):
    """[(W, b), ...] float64 numpy in forward order: hidden_0..hidden_{d-1}, readout.
    Mirrors swkprop/exact_meanprop extraction (W shape (out, in); b or None)."""
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


def run_spike_kprop(model, spike_dir="e1", input_dim: Optional[int] = None,
                    config: Optional[dict] = None, *, device: str = "cpu",
                    collect: bool = False) -> dict:
    """Predict ``E[model(X)]`` for ``X ~ N(0, I)`` by Spike K-Propagation along ``spike_dir``.

    ``spike_dir``: ``"e1"`` (localized), ``"ones"`` (flat all-ones), or a length-input_dim
    vector. ``config`` keys: ``R`` (2/3/4), ``n_nodes``, ``input_std``. ``device="cuda"`` routes
    the transverse congruence to the GPU (float64). Returns ``{"mean", "metadata", ...}``.
    """
    cfg = default_spike_kprop_config()
    if config:
        cfg.update(config)
    if model.cfg.activation != "relu":
        raise ValueError(f"SPIKE-KPROP supports ReLU only; got {model.cfg.activation!r}")
    if input_dim is None:
        input_dim = model.cfg.input_dim

    weights = _weights_from_model(model)
    mm = _make_mm(device)
    res = spike_kprop_predict(weights, input_dim, spike_dir, R=int(cfg["R"]),
                              n_nodes=int(cfg["n_nodes"]), input_std=float(cfg["input_std"]),
                              mm=mm, collect=collect)
    res["metadata"]["device"] = str(device)
    res["metadata"]["config"] = config_summary(spike_dir, config)
    return res


def extract_mean(result: dict) -> np.ndarray:
    return np.asarray(result["mean"], dtype=np.float64).reshape(-1)

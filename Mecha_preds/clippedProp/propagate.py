"""propagate.py -- run clippedProp forward through a study ``model.MLP``.

A ``model.MLP`` is ``depth`` hidden ``Linear -> activation`` blocks followed by a
linear readout. clippedProp walks the same structure, carrying a ``ClippedState``:

    for each hidden block:
        linear_layer            (pre-activation; scalar refit as plain Gaussian by default)
        [mean_subtraction_layer] if this block index is in ``mean_subtract_after``
        relu_layer              (the nonlinear, conditioning step)   -- skipped if identity
    linear_output_moments       (readout; EXACT, no closure)

Only ``relu`` and ``identity`` activations are handled (the project's models use
ReLU). Everything runs in float64 on CPU (the exact ReLU kernel is NumPy/SciPy).
"""
from __future__ import annotations

import torch

from .state import ClippedState, NEG_INF
from .layers import (
    linear_layer,
    linear_output_moments,
    mean_subtraction_layer,
    relu_layer,
)


@torch.no_grad()
def clipped_mlp_forward(
    model,
    *,
    init_state: ClippedState | None = None,
    input_dim: int | None = None,
    n_nodes: int = 21,
    relu_cov: str = "exact",
    clip_after_linear: bool = False,
    mean_subtract_after=(),
    want_cov: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Propagate the structured law through ``model`` and return the output moments.

    Args:
        model: a ``model.MLP`` (``.hidden_layers`` + ``.readout``, ReLU/identity).
        init_state: starting ``ClippedState``; defaults to ``X ~ N(0, I_input_dim)``.
        n_nodes: Gauss-Hermite nodes for the per-ReLU scalar integral.
        relu_cov: ``"exact"`` (bivariate, scipy) or ``"gain"`` (leading-order off-diag).
        clip_after_linear: refit the post-linear scalar as rectified (clamp 0)
            instead of a plain Gaussian (default False -- pre-activations are signed).
        mean_subtract_after: hidden-block indices after whose linear map to insert a
            mean-subtraction layer (for centered/LayerNorm-style models).
        want_cov: also return the full output covariance.

    Returns ``{"mean": Tensor(out_dim,), "cov": Tensor|None, "final_state": ClippedState}``.
    """
    activation = getattr(model.cfg, "activation", "relu")
    if activation not in ("relu", "identity"):
        raise ValueError(f"clippedProp supports relu/identity activations (got {activation!r})")

    if init_state is None:
        in_dim = input_dim if input_dim is not None else model.cfg.input_dim
        init_state = ClippedState.from_isotropic(in_dim, device=device, dtype=dtype)
    state = init_state

    lin_lo = 0.0 if clip_after_linear else NEG_INF
    ms_after = set(mean_subtract_after)

    for i, layer in enumerate(model.hidden_layers):
        b = layer.bias if layer.bias is not None else None
        state = linear_layer(state, layer.weight.to(dtype), None if b is None else b.to(dtype),
                             lo=lin_lo)
        if i in ms_after:
            state = mean_subtraction_layer(state)
        if activation == "relu":
            state = relu_layer(state, n_nodes=n_nodes, relu_cov=relu_cov)

    ro = model.readout
    rb = ro.bias if ro.bias is not None else None
    mu_out, Sigma_out = linear_output_moments(
        state, ro.weight.to(dtype), None if rb is None else rb.to(dtype))

    return {"mean": mu_out, "cov": Sigma_out if want_cov else None, "final_state": state}

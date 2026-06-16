"""polyjet.py -- scalar-h polynomial jets on GPU tensors.

A ``PolyJet`` represents a tensor-valued polynomial in the centered hidden mode
``dh = h - E[h]`` (q = 1, the scalar case of the spec's PolyJet):

    P(dh) = sum_{a=0}^p coeffs[a] * dh^a,     coeffs.shape = [p+1, *tensor_shape]

Coefficient-major layout (degree on axis 0, tensor indices after) so the linear
pushforward and expectation are single batched einsums (spec: "GPU coefficient
tensor layout"). For scalar h the polynomial product is a truncated convolution.

This mirrors the numpy oracle in ``_reference.py`` op-for-op; the package keeps
the torch path (GPU) and the numpy path (oracle) numerically identical.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class PolyJet:
    """Tensor-valued polynomial in scalar ``dh``; ``coeffs[a]`` multiplies ``dh^a``."""

    __slots__ = ("coeffs",)

    def __init__(self, coeffs: Tensor):
        if coeffs.ndim < 1:
            raise ValueError("coeffs must have a leading degree axis")
        self.coeffs = coeffs

    # -- constructors ----------------------------------------------------
    @classmethod
    def zeros(cls, p: int, shape, *, device=None, dtype=torch.float64) -> "PolyJet":
        return cls(torch.zeros((p + 1, *tuple(shape)), device=device, dtype=dtype))

    @classmethod
    def constant(cls, tensor: Tensor, p: int) -> "PolyJet":
        """Degree-0 jet: P(dh) = tensor (no hidden dependence)."""
        c = torch.zeros((p + 1, *tensor.shape), device=tensor.device, dtype=tensor.dtype)
        c[0] = tensor
        return cls(c)

    # -- structure -------------------------------------------------------
    @property
    def degree(self) -> int:
        return self.coeffs.shape[0] - 1

    @property
    def tensor_shape(self):
        return tuple(self.coeffs.shape[1:])

    @property
    def device(self):
        return self.coeffs.device

    @property
    def dtype(self):
        return self.coeffs.dtype

    def clone(self) -> "PolyJet":
        return PolyJet(self.coeffs.clone())

    # -- algebra ---------------------------------------------------------
    def add(self, other: "PolyJet") -> "PolyJet":
        p = max(self.degree, other.degree)
        out = torch.zeros((p + 1, *self.tensor_shape), device=self.device, dtype=self.dtype)
        out[: self.coeffs.shape[0]] += self.coeffs
        out[: other.coeffs.shape[0]] += other.coeffs
        return PolyJet(out)

    def sub(self, other: "PolyJet") -> "PolyJet":
        p = max(self.degree, other.degree)
        out = torch.zeros((p + 1, *self.tensor_shape), device=self.device, dtype=self.dtype)
        out[: self.coeffs.shape[0]] += self.coeffs
        out[: other.coeffs.shape[0]] -= other.coeffs
        return PolyJet(out)

    def scale(self, s) -> "PolyJet":
        return PolyJet(self.coeffs * s)

    def truncate(self, p: int) -> "PolyJet":
        if self.coeffs.shape[0] >= p + 1:
            return PolyJet(self.coeffs[: p + 1].clone())
        out = torch.zeros((p + 1, *self.tensor_shape), device=self.device, dtype=self.dtype)
        out[: self.coeffs.shape[0]] = self.coeffs
        return PolyJet(out)

    def evaluate_at_zero(self) -> Tensor:
        """The constant coefficient, P(0)."""
        return self.coeffs[0]

    def eval(self, t: float) -> Tensor:
        """Evaluate at scalar dh = t (Horner). Returns a tensor of tensor_shape."""
        acc = torch.zeros(self.tensor_shape, device=self.device, dtype=self.dtype)
        for a in range(self.coeffs.shape[0] - 1, -1, -1):
            acc = acc * t + self.coeffs[a]
        return acc

    def expectation(self, moment_vec: Tensor) -> Tensor:
        """E_h[P(dh)] = sum_a moment_vec[a] coeffs[a] (used for the tail diagnostic)."""
        m = moment_vec[: self.coeffs.shape[0]]
        return torch.tensordot(m, self.coeffs, dims=([0], [0]))


def jet_outer(P: PolyJet, Q: PolyJet, p: int) -> PolyJet:
    """(P outer Q) with convolution in degree: out[c,i,j] = sum_{a+b=c} P[a,i] Q[b,j].

    Truncated to degree p. P: [.,d_i], Q: [.,d_j] -> [p+1, d_i, d_j].
    """
    di = P.tensor_shape[0]
    dj = Q.tensor_shape[0]
    out = torch.zeros((p + 1, di, dj), device=P.device, dtype=P.dtype)
    for a in range(P.coeffs.shape[0]):
        for b in range(Q.coeffs.shape[0]):
            c = a + b
            if c <= p:
                out[c] += torch.outer(P.coeffs[a], Q.coeffs[b])
    return PolyJet(out)


def fit_jet(nodes: Tensor, values: Tensor, p: int, span: float) -> PolyJet:
    """Least-squares degree-p fit through (nodes, values[k, ...]) in the SCALED
    variable s = dh/span (nodes -> [-1, 1], well-conditioned Vandermonde), then
    convert to dh-monomial coefficients c_dh[a] = c_s[a] / span^a.
    """
    device, dtype = values.device, values.dtype
    s = (nodes / span).to(dtype)
    V = torch.stack([s ** a for a in range(p + 1)], dim=1)     # (M, p+1)
    flat = values.reshape(values.shape[0], -1)                 # (M, F)
    coef = torch.linalg.lstsq(V, flat).solution                # (p+1, F) in s
    scale = (1.0 / span) ** torch.arange(p + 1, device=device, dtype=dtype)
    coef = coef * scale[:, None]
    return PolyJet(coef.reshape((p + 1, *values.shape[1:])))

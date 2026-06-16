"""hidden.py -- scalar hidden-mode cumulants and the moments derived from them.

Stores the centered cumulants ``kappa_r`` (r >= 1, ``kappa_1 = 0``) of the scalar
hidden mode ``h`` and provides everything the marginalization / diagnostics need:

    moment(n)            E[dh^n] from cumulants (partition recursion)
    moment_vector(p)     [E[dh^0], ..., E[dh^p]]
    expect(jet)          E_h[P(dh)] = sum_a coeff[a] E[dh^a]
    contribution_by_degree(jet), gauss_hermite(n)   for diagnostics / marginalization

A Gaussian hidden mode is the special case ``kappa_r = 0`` for ``r != 2`` -- the
code never assumes it (spec design rule); ``is_gaussian()`` reports it so the
marginalizer can use the stable Gauss-Hermite path.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import torch
from torch import Tensor
from numpy.polynomial.hermite import hermgauss

SQRT2 = math.sqrt(2.0)


class HiddenCumulants:
    """Centered scalar hidden cumulants ``kappa[r]`` (index 0 unused, kappa[1]=0)."""

    def __init__(self, kappa: Sequence[float], *, device=None, dtype=torch.float64):
        self.kappa: List[float] = [float(k) for k in kappa]
        if len(self.kappa) >= 2:
            self.kappa[1] = 0.0                      # centered
        self.device = device
        self.dtype = dtype
        self._moment_cache = {0: 1.0, 1: 0.0}

    @classmethod
    def gaussian(cls, var: float = 1.0, p_hidden: int = 2, **kw) -> "HiddenCumulants":
        k = [0.0] * (p_hidden + 1)
        if p_hidden >= 2:
            k[2] = var
        return cls(k, **kw)

    @classmethod
    def none(cls, **kw) -> "HiddenCumulants":
        """No hidden mode (q = 0): all moments are E[dh^0]=1, E[dh^{>0}]=0."""
        return cls([0.0], **kw)

    # -- moments from cumulants -----------------------------------------
    def moment(self, n: int) -> float:
        """E[dh^n] via m_n = sum_{j=1..n} C(n-1,j-1) kappa_j m_{n-j}."""
        if n in self._moment_cache:
            return self._moment_cache[n]
        acc = 0.0
        for j in range(1, n + 1):
            kj = self.kappa[j] if j < len(self.kappa) else 0.0
            if kj != 0.0:
                acc += math.comb(n - 1, j - 1) * kj * self.moment(n - j)
        self._moment_cache[n] = acc
        return acc

    def moment_vector(self, p: int) -> Tensor:
        return torch.tensor([self.moment(a) for a in range(p + 1)],
                            device=self.device, dtype=self.dtype)

    def expect(self, jet) -> Tensor:
        return jet.expectation(self.moment_vector(jet.degree))

    @property
    def var(self) -> float:
        return self.kappa[2] if len(self.kappa) > 2 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 0.0))

    def is_gaussian(self, tol: float = 1e-12) -> bool:
        return all(abs(self.kappa[r]) <= tol
                   for r in range(1, len(self.kappa)) if r != 2)

    def has_mode(self) -> bool:
        return self.std > 0.0

    def gauss_hermite(self, n: int):
        """(nodes, weights) of ``h`` for E_{N(0,var)} via Gauss-Hermite quadrature."""
        t, om = hermgauss(n)
        nodes = torch.as_tensor(t, device=self.device, dtype=self.dtype) * SQRT2 * self.std
        weights = torch.as_tensor(om, device=self.device, dtype=self.dtype) / math.sqrt(math.pi)
        return nodes, weights

    # -- diagnostics -----------------------------------------------------
    def contribution_by_degree(self, jet) -> Tensor:
        """Per-degree moment-weighted coefficient norm (spec tail diagnostic)."""
        mv = self.moment_vector(jet.degree)
        flat = jet.coeffs.reshape(jet.coeffs.shape[0], -1)
        return (mv.abs() * flat.norm(dim=1))

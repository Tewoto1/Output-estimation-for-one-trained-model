"""state.py -- the clippedProp distribution state and the split/refit machinery.

We carry the law of a hidden vector ``X in R^d`` in the structured form

    X = s * u + z ,    u = 1_d / sqrt(d) ,    P = I - u u^T

with a CLAMPED-Gaussian scalar ``s`` along the all-ones direction ``u`` and a
Gaussian perpendicular part ``z`` in ``u^perp``, allowed to be correlated:

    s = max(lo, g),  g ~ N(m, v)                       (scalar channel; clamp ``lo``)
    z ~ N(mu_z, Sigma_z),   mu_z, Sigma_z in u^perp     (perpendicular channel)
    c_s = Cov(z, s) in u^perp                           (cross-covariance)

The underlying ``(g, z)`` is jointly Gaussian; ``s`` is the clamp of the scalar
latent ``g``. Storing ``Cov(z, s)`` (clamped) keeps the linear-layer covariance
assembly direct; the ReLU step converts it to ``Cov(z, g) = c_s / beta`` for
Gaussian conditioning (see ``scalar.clipped_cross_beta``).

The two reconstructions both routes need:

    mean_cov()           : (m, v, lo, mu_z, Sigma_z, c_s)  ->  full (mu_X, Sigma_X)
    from_full_moments()  : full (mu_Y, Sigma_Y)            ->  state   ("re-split + refit")

``from_full_moments`` is the projection (closure) applied after every layer: it
splits the layer-output moments onto ``u`` / ``u^perp`` and refits the scalar to
a clamped Gaussian, possibly in a NEW dimension (linear layers change width).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .scalar import rect_gauss_moments, fit_rect_gauss

NEG_INF = -math.inf


# ---------------------------------------------------------------------------
# Projection helpers for u = 1/sqrt(d) (no need to materialize P)
# ---------------------------------------------------------------------------
def _u(d: int, *, device=None, dtype=torch.float64) -> Tensor:
    return torch.ones(d, device=device, dtype=dtype) / math.sqrt(d)


def proj_vec(x: Tensor, u: Tensor) -> Tensor:
    """P x = x - (u . x) u."""
    return x - (u @ x) * u


def proj_mat(S: Tensor, u: Tensor) -> Tensor:
    """P S P = S - u (S u)^T - (S u) u^T + (u^T S u) u u^T  (kept symmetric)."""
    Su = S @ u
    uSu = u @ Su
    out = S - torch.outer(u, Su) - torch.outer(Su, u) + uSu * torch.outer(u, u)
    return 0.5 * (out + out.T)


@dataclass
class ClippedState:
    """Structured law of one hidden vector: clamped scalar ``s`` on ``u`` + Gaussian ``z`` on ``u^perp``."""
    d: int
    m: float                 # scalar latent mean        (g ~ N(m, v))
    v: float                 # scalar latent variance
    lo: float                # scalar clamp lower bound  (0 => rectified; -inf => plain Gaussian)
    mu_z: Tensor             # (d,)  perpendicular mean        (in u^perp)
    Sigma_z: Tensor          # (d,d) perpendicular covariance  (in u^perp)
    c_s: Tensor              # (d,)  Cov(z, s)                 (in u^perp)

    # -- basic accessors ----------------------------------------------------
    @property
    def device(self):
        return self.mu_z.device

    @property
    def dtype(self):
        return self.mu_z.dtype

    @property
    def u(self) -> Tensor:
        return _u(self.d, device=self.device, dtype=self.dtype)

    def scalar_moments(self) -> tuple[float, float, float]:
        """``(E[s], Var[s], p0)`` of the clamped-Gaussian scalar channel."""
        return rect_gauss_moments(self.m, self.v, self.lo)

    # -- full (mean, covariance) reconstruction -----------------------------
    def mean_cov(self) -> tuple[Tensor, Tensor]:
        """Reconstruct the full first two moments of ``X = s u + z``::

            mu_X    = E[s] u + mu_z
            Sigma_X = Var[s] u u^T + u c_s^T + c_s u^T + Sigma_z
        """
        u = self.u
        e_s, var_s, _ = self.scalar_moments()
        mu_X = e_s * u + self.mu_z
        Sigma_X = (var_s * torch.outer(u, u)
                   + torch.outer(u, self.c_s) + torch.outer(self.c_s, u)
                   + self.Sigma_z)
        return mu_X, 0.5 * (Sigma_X + Sigma_X.T)

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_isotropic(cls, d: int, *, device=None, dtype=torch.float64) -> "ClippedState":
        """``X ~ N(0, I_d)``: scalar ``s = u^T X ~ N(0, 1)`` (plain Gaussian),
        ``z ~ N(0, P)``, ``Cov(z, s) = P u = 0``."""
        u = _u(d, device=device, dtype=dtype)
        Sigma_z = torch.eye(d, device=device, dtype=dtype) - torch.outer(u, u)
        zeros = torch.zeros(d, device=device, dtype=dtype)
        return cls(d=d, m=0.0, v=1.0, lo=NEG_INF, mu_z=zeros, Sigma_z=Sigma_z, c_s=zeros.clone())

    @classmethod
    def from_gaussian(cls, mu: Tensor, Sigma: Tensor) -> "ClippedState":
        """Split an arbitrary Gaussian ``N(mu, Sigma)`` -- the scalar channel of a
        Gaussian is itself Gaussian (``lo = -inf``)."""
        return cls.from_full_moments(mu, Sigma, lo=NEG_INF)

    @classmethod
    def from_structured(cls, d: int, m: float, v: float, *, lo: float = 0.0,
                        mu_z: Tensor | None = None, Sigma_z: Tensor | None = None,
                        c_s: Tensor | None = None, device=None,
                        dtype=torch.float64) -> "ClippedState":
        """Build the user's structured input directly: a clamped-Gaussian scalar
        ``s = max(lo, N(m, v))`` on ``u`` plus a Gaussian ``z`` on ``u^perp``.

        ``mu_z`` / ``Sigma_z`` / ``c_s`` default to zero and are re-projected into
        ``u^perp`` so the stored state is consistent.
        """
        u = _u(d, device=device, dtype=dtype)
        mu_z = torch.zeros(d, device=device, dtype=dtype) if mu_z is None else mu_z.to(dtype)
        Sigma_z = (torch.zeros(d, d, device=device, dtype=dtype)
                   if Sigma_z is None else Sigma_z.to(dtype))
        c_s = torch.zeros(d, device=device, dtype=dtype) if c_s is None else c_s.to(dtype)
        return cls(d=d, m=float(m), v=float(v), lo=float(lo),
                   mu_z=proj_vec(mu_z, u), Sigma_z=proj_mat(Sigma_z, u), c_s=proj_vec(c_s, u))

    # -- the closure: re-split layer-output moments + refit the scalar ------
    @classmethod
    def from_full_moments(cls, mu_Y: Tensor, Sigma_Y: Tensor, *, lo: float) -> "ClippedState":
        """Project full moments ``(mu_Y, Sigma_Y)`` onto the structured state.

        Splits onto the (possibly NEW) all-ones direction ``u' = 1/sqrt(d')`` and
        its complement, then refits the scalar channel to a clamped Gaussian with
        clamp ``lo``::

            E[s'] = u'^T mu_Y      Var[s'] = u'^T Sigma_Y u'
            mu_z' = P' mu_Y        Sigma_z' = P' Sigma_Y P'      c_s' = P' Sigma_Y u'
            (m', v') = fit_rect_gauss(E[s'], Var[s'], lo)
        """
        mu_Y = mu_Y.reshape(-1)
        d = mu_Y.shape[0]
        Sigma_Y = 0.5 * (Sigma_Y + Sigma_Y.T)
        u = _u(d, device=mu_Y.device, dtype=mu_Y.dtype)

        Su = Sigma_Y @ u
        e_s = float(u @ mu_Y)
        var_s = max(float(u @ Su), 0.0)
        mu_z = proj_vec(mu_Y, u)
        Sigma_z = proj_mat(Sigma_Y, u)
        c_s = Su - (u @ Su) * u                      # P' Sigma_Y u'
        m, v = fit_rect_gauss(e_s, var_s, lo)
        return cls(d=d, m=m, v=v, lo=float(lo), mu_z=mu_z, Sigma_z=Sigma_z, c_s=c_s)

    # -- diagnostics --------------------------------------------------------
    def summary(self) -> dict:
        e_s, var_s, p0 = self.scalar_moments()
        return {
            "d": self.d,
            "scalar_latent": {"m": self.m, "v": self.v, "lo": self.lo},
            "scalar_moments": {"mean": e_s, "var": var_s, "p0": p0},
            "mu_z_norm": float(self.mu_z.norm()),
            "Sigma_z_trace": float(torch.diag(self.Sigma_z).sum()),
            "c_s_norm": float(self.c_s.norm()),
        }

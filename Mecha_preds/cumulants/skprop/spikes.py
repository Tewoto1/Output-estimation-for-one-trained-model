"""spikes.py -- find the low-rank structured ("spiked"/meaned) part of a weight matrix.

Structured power KPROP needs to know WHICH directions carry the low-rank latent.
This module supplies them: ``detect_spikes(W)`` runs a sequential
Marchenko--Pastur edge test on the singular values of ``W`` and returns the
outlier singular triplets ``(u_k, s_k, v_k)`` -- the ``A = W_random + U S V^T``
decomposition of the writeup, found from ``W`` alone (no init needed).

Test: after provisionally removing the top ``k`` singular values, the residual
entry variance is ``sigma_k^2 ~= sum_{i>k} s_i^2 / (m*n - k*(m+n))`` and an
i.i.d. bulk cannot produce singular values above ``sigma_k * (sqrt(m)+sqrt(n))``
(the MP bulk edge). ``s_k > margin * edge_k`` => spike. Rank is capped at
``q_max`` because each latent multiplies the quadrature cost of the structured
algorithm by ``n_nodes``.

Callers that already KNOW the structure (an all-ones meaned matrix, the -mu
direction from the Q2 study) should skip detection and pass explicit
``directions=`` to ``structured_mlp_kprop`` instead -- detection is the
fallback, not the source of truth.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SpikeInfo:
    """Low-rank structured part of one weight matrix: ``W ~ bulk + U diag(s) V^T``."""
    layer: int                  # which linear layer (index into mlp.Ws); -1 = not layer-tagged
    q: int                      # number of detected spikes (0 = pure bulk)
    U: Tensor                   # (out_dim, q) left singular vectors (output-side directions)
    s: Tensor                   # (q,) outlier singular values
    V: Tensor                   # (in_dim, q) right singular vectors (latent directions)
    sigma_hat: float            # estimated bulk entry std after removing the spikes
    bulk_edge: float            # MP bulk edge sigma_hat*(sqrt(m)+sqrt(n))
    all_sv: Tensor = field(default_factory=lambda: torch.empty(0))  # full spectrum (diagnostics)

    def summary(self) -> dict:
        return {
            "layer": self.layer,
            "q": self.q,
            "s": [float(x) for x in self.s],
            "sigma_hat": self.sigma_hat,
            "bulk_edge": self.bulk_edge,
            "overshoot": [float(x) / self.bulk_edge for x in self.s] if self.bulk_edge > 0 else [],
        }


def detect_spikes(
    W: Tensor,
    *,
    q_max: int = 2,
    margin: float = 1.15,
    layer: int = -1,
) -> SpikeInfo:
    """Sequential MP-edge test for outlier singular values of ``W`` (out_dim, in_dim).

    Args:
        q_max: cap on the detected rank (each spike costs a quadrature dimension).
        margin: a singular value must exceed ``margin * bulk_edge`` to count.
            >1 keeps borderline bulk fluctuations (which the vanilla algorithm
            already handles) from triggering needless conditioning.
    Returns:
        SpikeInfo with q in [0, q_max]. q=0 means "no structure found": the
        structured algorithm then degenerates to vanilla kprop exactly.
    """
    W = torch.as_tensor(W)
    m, n = W.shape
    U_full, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    total = float(S.pow(2).sum())

    q = 0
    sigma_hat = math.sqrt(total / (m * n))
    edge = sigma_hat * (math.sqrt(m) + math.sqrt(n))
    for k in range(min(q_max, len(S))):
        resid = total - float(S[: k + 1].pow(2).sum())
        dof = max(m * n - (k + 1) * (m + n), 1)
        sigma_k = math.sqrt(max(resid, 0.0) / dof)
        edge_k = sigma_k * (math.sqrt(m) + math.sqrt(n))
        if float(S[k]) > margin * edge_k:
            q = k + 1
            sigma_hat, edge = sigma_k, edge_k
        else:
            break

    return SpikeInfo(
        layer=layer,
        q=q,
        U=U_full[:, :q].to(W.dtype),
        s=S[:q].to(W.dtype),
        V=Vh[:q, :].T.contiguous().to(W.dtype),
        sigma_hat=sigma_hat,
        bulk_edge=edge,
        all_sv=S.to(W.dtype),
    )


def detect_spikes_all_layers(
    Ws: List[Tensor],
    *,
    q_max: int = 2,
    margin: float = 1.15,
) -> List[SpikeInfo]:
    """Run ``detect_spikes`` on every weight matrix (diagnostics / deep mode)."""
    return [detect_spikes(W, q_max=q_max, margin=margin, layer=l) for l, W in enumerate(Ws)]


def orthonormalize(directions: Tensor) -> Tensor:
    """QR-orthonormalize explicit latent directions, shape (dim, q) (or (dim,) -> (dim, 1))."""
    directions = torch.as_tensor(directions)
    if directions.ndim == 1:
        directions = directions[:, None]
    Q, R = torch.linalg.qr(directions.double())
    # Drop numerically dependent columns
    keep = R.diagonal().abs() > 1e-10
    if not bool(keep.all()):
        logger.warning("orthonormalize: dropping %d dependent direction(s)", int((~keep).sum()))
    return Q[:, keep].to(directions.dtype)

"""_cumulant_math.py -- shared probabilists'-Hermite and moment<->cumulant helpers.

Single home for the small numpy-only algebra reused by the Edgeworth / Gram-Charlier ReLU
steps of the torch-free predictors (``swkprop``, ``spikekprop``). Kept here so the identical
formulas are written once, not copied per predictor.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def He(p: int, z: np.ndarray) -> np.ndarray:
    """Probabilists' Hermite polynomial ``He_p(z)`` (orders 0..6)."""
    if p == 0:
        return np.ones_like(z)
    if p == 1:
        return z
    if p == 2:
        return z * z - 1.0
    if p == 3:
        return z ** 3 - 3.0 * z
    if p == 4:
        return z ** 4 - 6.0 * z ** 2 + 3.0
    if p == 5:
        return z ** 5 - 10.0 * z ** 3 + 15.0 * z
    if p == 6:
        return z ** 6 - 15.0 * z ** 4 + 45.0 * z ** 2 - 15.0
    raise NotImplementedError(f"He_{p} not implemented (R<=6 supported)")


def central_moments_to_cumulants(mu_c: Dict[int, float], max_p: int) -> Dict[int, float]:
    """Central moments ``{2:mu2, 3:mu3, ...}`` -> cumulants ``{p: kappa_p}`` for p=2..max_p."""
    k: Dict[int, float] = {}
    m2 = mu_c.get(2, 0.0)
    k[2] = m2
    if max_p >= 3:
        k[3] = mu_c.get(3, 0.0)
    if max_p >= 4:
        k[4] = mu_c.get(4, 0.0) - 3.0 * m2 * m2
    if max_p >= 5:
        k[5] = mu_c.get(5, 0.0) - 10.0 * mu_c.get(3, 0.0) * m2
    if max_p >= 6:
        k[6] = (mu_c.get(6, 0.0) - 15.0 * mu_c.get(4, 0.0) * m2
                - 10.0 * mu_c.get(3, 0.0) ** 2 + 30.0 * m2 ** 3)
    return k

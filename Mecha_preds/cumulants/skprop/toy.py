"""toy.py -- the conditional-Gaussian toy model with EXACT closed forms.

The minimal model where a meaned matrix breaks vanilla kprop and the
structured-latent + power-cumulant split provably fixes it:

    P_i = a_i H + sigma G_i,   H, G_1..G_n ~ N(0,1) iid     (shared latent H)
    X_i = phi(P_i) = P_i^2                                   (square activation)
    M   = (1/n) sum_i X_i                                    (the meaned channel)
    target: E[(lam*M)^3]                                     (cubic readout)

Conditional on H, X_i = (a_i H + sigma G_i)^2 is a scaled noncentral chi-square
with EXACT cumulants (kappa_r = 2^{r-1}(r-1)! sigma^{2r} (1 + r mu_i^2/sigma^2),
mu_i = a_i H):

    kappa_1[X_i|H] = a_i^2 H^2 + sigma^2
    kappa_2[X_i|H] = 2 sigma^4 + 4 sigma^2 a_i^2 H^2
    kappa_3[X_i|H] = 8 sigma^6 + 24 sigma^4 a_i^2 H^2

and conditional independence gives kappa_s[delta|H] = n^{-s} sum_i kappa_s[X_i|H]
for delta = M - m(H), m(H) = E[M|H]. Orders 2, 3 cumulants equal central
moments, so

    E[M^3] = E[m(H)^3] + 3 E[m(H) kappa_2[delta|H]] + E[kappa_3[delta|H]]   (EXACT).

The four algorithms of the writeup, in increasing structure-awareness:

    A  vanilla Gaussian-ish kprop: correct mean, but variance/skewness from the
       DIAGONAL only (coherent off-diagonal Cov(X_i,X_j)=2a_i^2a_j^2 H-latent
       structure dropped)                                   -> error O(1)
    B  track the latent m(H), ignore finite-width noise      -> error O(1/n)
    C  B + conditional power cumulant kappa_2[delta|H]       -> error O(1/n^2)
    D  C + kappa_3[delta|H]                                  -> exact (0 up to fp)

Everything is computed exactly by representing functions of H as polynomials in
H and integrating with E[H^{2k}] = (2k-1)!!. ``mc_EM3`` cross-checks by sampling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def gaussian_even_moments(max_pow2: int) -> np.ndarray:
    """[E[H^0], E[H^2], ..., E[H^{2*max_pow2}]] = (2k-1)!! for H ~ N(0,1)."""
    out = np.ones(max_pow2 + 1)
    for k in range(1, max_pow2 + 1):
        out[k] = out[k - 1] * (2 * k - 1)
    return out


def _e_poly_h2(coefs: np.ndarray) -> float:
    """E[sum_k coefs[k] * H^{2k}] for H ~ N(0,1)."""
    return float(np.dot(coefs, gaussian_even_moments(len(coefs) - 1)))


def _pmul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Multiply two polynomials in H^2 (coefficient arrays over H^{2k})."""
    return np.convolve(p, q)


@dataclass
class ToyModel:
    """P_i = a_i H + sigma G_i, phi = square, meaned channel M, readout (lam*M)^3."""
    a: np.ndarray          # (n,) latent loadings
    sigma: float           # coordinate noise scale
    lam: float = 1.0       # meaned-matrix strength: Y = lam * M

    def __post_init__(self):
        self.a = np.asarray(self.a, dtype=np.float64)

    @property
    def n(self) -> int:
        return len(self.a)

    # -- conditional pieces, as polynomials in H^2 ------------------------
    def m_poly(self) -> np.ndarray:
        """m(H) = E[M|H] = abar2 * H^2 + sigma^2 -> [sigma^2, abar2]."""
        return np.array([self.sigma**2, float(np.mean(self.a**2))])

    def kappa2_delta_poly(self) -> np.ndarray:
        """kappa_2[delta|H] = 2 sigma^4 / n + (4 sigma^2 / n^2) (sum a_i^2) H^2."""
        n, s2 = self.n, self.sigma**2
        return np.array([2 * s2**2 / n, 4 * s2 * float(np.sum(self.a**2)) / n**2])

    def kappa3_delta_poly(self) -> np.ndarray:
        """kappa_3[delta|H] = 8 sigma^6 / n^2 + (24 sigma^4 / n^3) (sum a_i^2) H^2."""
        n, s2 = self.n, self.sigma**2
        return np.array([8 * s2**3 / n**2, 24 * s2**2 * float(np.sum(self.a**2)) / n**3])

    # -- exact target ------------------------------------------------------
    def exact_EM3(self) -> float:
        """E[M^3] = E[m^3] + 3 E[m kappa_2] + E[kappa_3] (exact identity here)."""
        m = self.m_poly()
        m3 = _pmul(_pmul(m, m), m)
        return (_e_poly_h2(m3)
                + 3.0 * _e_poly_h2(_pmul(m, self.kappa2_delta_poly()))
                + _e_poly_h2(self.kappa3_delta_poly()))

    def exact_target(self) -> float:
        return self.lam**3 * self.exact_EM3()

    # -- the four algorithms ------------------------------------------------
    def estimates(self) -> Dict[str, float]:
        """E_hat[(lam M)^3] under algorithms A, B, C, D (see module docstring)."""
        n = self.n
        m = self.m_poly()
        m3 = _pmul(_pmul(m, m), m)

        # A: unconditional mean exact; var/skewness from per-coordinate
        # (diagonal) cumulants only. Unconditionally P_i ~ N(0, a_i^2+sigma^2),
        # X_i = P_i^2: Var = 2 v_i^2, kappa_3 = 8 v_i^3, v_i = a_i^2 + sigma^2.
        v = self.a**2 + self.sigma**2
        mu_M = float(np.mean(v))
        var_diag = 2.0 * float(np.sum(v**2)) / n**2
        k3_diag = 8.0 * float(np.sum(v**3)) / n**3
        EM3_A = mu_M**3 + 3 * mu_M * var_diag + k3_diag

        EM3_B = _e_poly_h2(m3)
        EM3_C = EM3_B + 3.0 * _e_poly_h2(_pmul(m, self.kappa2_delta_poly()))
        EM3_D = EM3_C + _e_poly_h2(self.kappa3_delta_poly())
        lam3 = self.lam**3
        return {"A": lam3 * EM3_A, "B": lam3 * EM3_B, "C": lam3 * EM3_C, "D": lam3 * EM3_D}

    def errors(self) -> Dict[str, float]:
        exact = self.exact_target()
        return {k: abs(v - exact) for k, v in self.estimates().items()}

    # -- Monte-Carlo cross-check --------------------------------------------
    def mc_EM3(self, n_samples: int = 1_000_000, seed: int = 0,
               batch: int = 200_000) -> float:
        rng = np.random.default_rng(seed)
        total, done = 0.0, 0
        while done < n_samples:
            b = min(batch, n_samples - done)
            H = rng.standard_normal(b)[:, None]
            G = rng.standard_normal((b, self.n))
            M = np.mean((self.a[None, :] * H + self.sigma * G) ** 2, axis=1)
            total += float(np.sum(M**3))
            done += b
        return self.lam**3 * total / n_samples


def make_toy(n: int, *, a_scale: float = 1.0, sigma: float = 1.0, lam: float = 1.0,
             seed: int = 0, iid_loadings: bool = True) -> ToyModel:
    """Standard instance: a_i = a_scale * (1 + 0.5 z_i)/norm so abar2 stays O(1) in n."""
    rng = np.random.default_rng(seed)
    a = a_scale * (1.0 + 0.5 * rng.standard_normal(n)) if iid_loadings \
        else a_scale * np.ones(n)
    return ToyModel(a=a, sigma=sigma, lam=lam)


def error_sweep(ns, *, a_scale: float = 1.0, sigma: float = 1.0, lam: float = 1.0,
                seed: int = 0) -> Dict[str, np.ndarray]:
    """abs error of A/B/C/D vs n -- the O(1), O(1/n), O(1/n^2), ~0 hierarchy."""
    ns = np.asarray(list(ns))
    out = {k: np.zeros(len(ns)) for k in "ABCD"}
    out["n"] = ns.astype(float)
    out["exact"] = np.zeros(len(ns))
    for i, n in enumerate(ns):
        toy = make_toy(int(n), a_scale=a_scale, sigma=sigma, lam=lam, seed=seed)
        out["exact"][i] = toy.exact_target()
        for k, err in toy.errors().items():
            out[k][i] = err
    return out

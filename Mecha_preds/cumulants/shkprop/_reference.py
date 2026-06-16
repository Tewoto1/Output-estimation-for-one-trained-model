"""_reference.py -- pure-numpy ORACLE for SCALAR hidden-mode kprop.

The torch modules in this package mirror this file function-for-function. The
numpy version is the spec-correctness reference (no torch dependency, so it runs
anywhere) and a permanent regression oracle: the notebook and tests cross-check
the torch path against ``symbolic_kprop`` here and assert agreement to ~1e-12.
Import as ``from Mecha_preds.cumulants.shkprop import reference as ref``.

Scalar hidden mode h (q=1). State per layer = conditional cumulants of the layer
activations A | h, stored as truncated polynomial jets in dh = h - E[h]:

    K1(h) ~= sum_{a=0..p} K1[a] dh^a            coeffs shape [p+1, d]
    K2(h) ~= sum_{a=0..p} K2[a] dh^a            coeffs shape [p+1, d, d]

k_max = 2 (mean + covariance). ReLU activation under the conditional-Gaussian
residual closure (Z|h ~ N(mu(h), Cov(h))). Everything here is numpy.
"""
import math
import numpy as np
from numpy.polynomial.hermite import hermgauss

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
_erf = np.vectorize(math.erf)


def npdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / SQRT2PI


def ncdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x) / SQRT2))


# ---------------------------------------------------------------------------
# Scalar-h polynomial jets (axis 0 = degree, remaining axes = tensor shape)
# ---------------------------------------------------------------------------
def jet_zeros(p, shape):
    return np.zeros((p + 1,) + tuple(shape), dtype=np.float64)


def jet_truncate(P, p):
    if P.shape[0] >= p + 1:
        return P[: p + 1].copy()
    out = np.zeros((p + 1,) + P.shape[1:], dtype=P.dtype)
    out[: P.shape[0]] = P
    return out


def jet_eval(P, t):
    """Evaluate jet at scalar dh=t (Horner). Returns tensor of shape P.shape[1:]."""
    acc = np.zeros(P.shape[1:], dtype=np.float64)
    for a in range(P.shape[0] - 1, -1, -1):
        acc = acc * t + P[a]
    return acc


def jet_outer(P, Q, p):
    """(P outer Q) with convolution in degree: out[c,i,j]=sum_{a+b=c} P[a,i] Q[b,j].

    P: [.,d_i], Q: [.,d_j] -> [p+1, d_i, d_j]. Truncated to degree p.
    """
    di, dj = P.shape[1], Q.shape[1]
    out = np.zeros((p + 1, di, dj), dtype=np.float64)
    for a in range(P.shape[0]):
        for b in range(Q.shape[0]):
            c = a + b
            if c <= p:
                out[c] += np.multiply.outer(P[a], Q[b])
    return out


def jet_expect(P, moment_vec):
    """E_h[P(dh)] = sum_a moment_vec[a] P[a]. moment_vec must cover deg(P)."""
    m = moment_vec[: P.shape[0]]
    return np.tensordot(m, P, axes=([0], [0]))


# ---------------------------------------------------------------------------
# Hidden cumulants (scalar, centered): moments from cumulants
# ---------------------------------------------------------------------------
def moments_from_cumulants(kappa, n_max):
    """kappa[r] for r>=1 (kappa[1]=0 centered). Returns m[0..n_max], m_n=E[dh^n].

    Recursion m_n = sum_{j=1..n} C(n-1,j-1) kappa_j m_{n-j}.
    """
    m = np.zeros(n_max + 1, dtype=np.float64)
    m[0] = 1.0
    for n in range(1, n_max + 1):
        acc = 0.0
        for j in range(1, n + 1):
            kj = kappa[j] if j < len(kappa) else 0.0
            acc += math.comb(n - 1, j - 1) * kj * m[n - j]
        m[n] = acc
    return m


def gaussian_kappa(p_hidden, var=1.0):
    """Cumulants of a centered Gaussian: kappa_2=var, rest 0. Index 0 unused."""
    k = [0.0] * (p_hidden + 1)
    if p_hidden >= 2:
        k[2] = var
    return k


# ---------------------------------------------------------------------------
# ReLU-Gaussian conditional moments (the pointwise scalar formulas)
# ---------------------------------------------------------------------------
def relu_m1(mu, var, eps=1e-12):
    """E[ReLU(Z)], Z~N(mu, var). Elementwise over arrays."""
    sig = np.sqrt(np.maximum(var, eps))
    a = mu / sig
    return sig * npdf(a) + mu * ncdf(a)


def relu_m2_diag(mu, var, eps=1e-12):
    """E[ReLU(Z)^2], Z~N(mu,var). Elementwise."""
    sig = np.sqrt(np.maximum(var, eps))
    a = mu / sig
    return (mu ** 2 + var) * ncdf(a) + mu * sig * npdf(a)


def relu_pair_matrix(mu, C, n_gl=64, span=12.0, eps=1e-12):
    """E[ReLU(Z_i) ReLU(Z_j)] for all i,j with Z~N(mu, C). Returns (d,d), symmetric.

    Reduction (exact for the inner variable, quadrature only on the outer one):
    write Z_i = mu_i + s_i u, Z_j = mu_j + L10_ij u + L11_ij v with u,v iid N(0,1)
    (elementwise 2x2 Cholesky: s_i=sqrt(C_ii), L10_ij=C_ij/s_i, L11_ij^2=C_jj-L10_ij^2).
    Integrate v in closed form -> E_v[ReLU(Z_j)|u] = relu_m1(mu_j + L10_ij u, L11_ij^2).
    What remains,  E[X_i^+ X_j^+] = int_{u0_i}^inf (mu_i + s_i u) phi(u)
                                       relu_m1(mu_j + L10_ij u, L11_ij^2) du,
    has a SMOOTH integrand on the half-line past the ReLU kink u0_i = -mu_i/s_i, so a
    half-line Gauss-Legendre rule is near-exact (the kink that defeats Gauss-Hermite
    is gone). Validated to ~1e-4 (MC-floor) incl. correlations up to ~0.91.
    """
    d = mu.shape[0]
    vi = np.maximum(np.diag(C), eps)
    si = np.sqrt(vi)                                   # (d,)  = s_i
    L10 = C / si[:, None]                              # (d,d) uses s_i on row i
    L11sq = np.maximum(vi[None, :] - L10 ** 2, 0.0)    # (d,d) = L11_ij^2
    u0 = -mu / si                                      # (d,)  per-row ReLU kink
    g, gw = np.polynomial.legendre.leggauss(n_gl)      # on [-1, 1]
    nodes = 0.5 * (g + 1.0) * span                     # half-line [0, span]
    wts = 0.5 * span * gw
    out = np.zeros((d, d), dtype=np.float64)
    for k in range(n_gl):
        u = u0[:, None] + nodes[k]                     # (d,1) u >= u0_i (smooth region)
        lin = mu[:, None] + si[:, None] * u            # (d,1) = ReLU(Z_i) value (>=0 here)
        m1j = relu_m1(mu[None, :] + L10 * u, L11sq, eps=eps)  # (d,d) inner E_v[ReLU(Z_j)]
        out += wts[k] * lin * npdf(u) * m1j
    out = 0.5 * (out + out.T)                          # symmetric by construction
    di = np.arange(d)
    out[di, di] = relu_m2_diag(mu, np.diag(C), eps=eps)  # exact diagonal
    return out


# ---------------------------------------------------------------------------
# Collocation composition over scalar dh (pseudo-spectral; exact for poly<=p)
# ---------------------------------------------------------------------------
def collocation_nodes(p, span=4.0, oversample=3):
    """M=oversample*p+1 distinct dh nodes (Chebyshev extrema on [-span,span]).

    span ~4 covers the +-4 sigma bulk of a standardized Gaussian latent.
    """
    M = oversample * p + 1
    if M == 1:
        return np.array([0.0])
    k = np.arange(M)
    return span * np.cos(math.pi * k / (M - 1))        # Chebyshev extrema


def fit_jet(nodes, values, p, span=4.0):
    """Least-squares degree-p fit through (nodes, values[k, ...]).

    Fit in the SCALED variable s = dh/span (nodes -> [-1, 1]) so the Vandermonde is
    well-conditioned, then convert to dh-monomial coefficients c_dh[a] = c_s[a]/span^a
    (float64 high-degree-stability guidance from the spec).
    """
    s = nodes / span
    V = np.vander(s, p + 1, increasing=True)           # (M, p+1) in [-1,1]
    flat = values.reshape(values.shape[0], -1)         # (M, F)
    coef, *_ = np.linalg.lstsq(V, flat, rcond=None)    # (p+1, F) in s
    scale = (1.0 / span) ** np.arange(p + 1)           # c_dh[a] = c_s[a] / span^a
    coef = coef * scale[:, None]
    return coef.reshape((p + 1,) + values.shape[1:])


# ---------------------------------------------------------------------------
# Layer updates
# ---------------------------------------------------------------------------
def linear_pushforward(K1, K2, W, b=None):
    """Exact: K1_out[a,o]=sum_i W[o,i]K1[a,i]; K2_out[a,o,p]=W K2[a] W^T."""
    K1o = np.einsum("oi,ai->ao", W, K1)
    if b is not None:
        K1o[0] = K1o[0] + b
    K2o = np.einsum("oi,aij,pj->aop", W, K2, W)
    return K1o, K2o


def activation_relu(K1, K2, p, n_gl=64, var_floor=1e-12):
    """ReLU conditional-Gaussian closure as a jet update (k_max=2).

    Builds raw-moment jets E[ReLU|h], E[ReLU ReLU^T|h] by collocation, then
    converts to cumulant jets: K1=raw1, K2=raw2 - raw1 outer raw1.
    """
    d = K1.shape[1]
    nodes = collocation_nodes(p)
    raw1_vals = np.zeros((len(nodes), d), dtype=np.float64)
    raw2_vals = np.zeros((len(nodes), d, d), dtype=np.float64)
    for k, t in enumerate(nodes):
        mu = jet_eval(K1, t)                            # (d,)
        C = jet_eval(K2, t)                             # (d,d)
        C = 0.5 * (C + C.T)
        var = np.maximum(np.diag(C), var_floor)
        raw1_vals[k] = relu_m1(mu, var, eps=var_floor)
        raw2_vals[k] = relu_pair_matrix(mu, C, n_gl=n_gl, eps=var_floor)
    raw1 = fit_jet(nodes, raw1_vals, p)                 # [p+1, d]
    raw2 = fit_jet(nodes, raw2_vals, p)                 # [p+1, d, d]
    K1o = raw1
    K2o = raw2 - jet_outer(raw1, raw1, p)               # raw -> cumulant (k=2)
    return K1o, K2o


def tail_score(K1, moment_vec, band=1):
    """Contribution-by-degree tail score of the mean jet (diagnostic)."""
    contrib = np.array([
        np.linalg.norm(moment_vec[a] * K1[a]) for a in range(K1.shape[0])
    ])
    total = contrib.sum() + 1e-30
    p = K1.shape[0] - 1
    tail = contrib[max(0, p - band + 1):].sum()
    return tail / total


def is_gaussian(kappa, tol=1e-12):
    """True if hidden cumulants are Gaussian (kappa_r ~ 0 for r != 2)."""
    return all(abs(kappa[r]) <= tol for r in range(1, len(kappa)) if r != 2)


def marginalize(K1, K2, kappa, n_quad=None):
    """Unconditional mean & covariance via law of total covariance.

    Gaussian h (the validated path): marginalize by GAUSS-HERMITE QUADRATURE -- evaluate
    the jets at GH nodes h_k (|h_k| <~ 4 sigma) and take the weighted sum. This is exact
    for polynomials up to degree 2*n_quad-1 and avoids the float64 cancellation of summing
    coeff * E[dh^a] with rapidly growing high moments. Non-Gaussian h falls back to moments.
    """
    p = K1.shape[0] - 1
    if is_gaussian(kappa):
        std = math.sqrt(kappa[2]) if len(kappa) > 2 else 0.0
        if n_quad is None:
            n_quad = max(p + 4, 8)
        if std == 0.0:                                      # no hidden mode (q=0)
            return jet_eval(K1, 0.0), jet_eval(K2, 0.0)
        t, om = hermgauss(n_quad)
        nodes, w = t * SQRT2 * std, om / math.sqrt(math.pi)
        mean = np.zeros(K1.shape[1:]); E_cov = np.zeros(K2.shape[1:])
        E_mmT = np.zeros(K2.shape[1:])
        for hk, wk in zip(nodes, w):
            m_h = jet_eval(K1, hk)                           # (d,)
            mean = mean + wk * m_h
            E_cov = E_cov + wk * jet_eval(K2, hk)
            E_mmT = E_mmT + wk * np.outer(m_h, m_h)
        cov = E_cov + (E_mmT - np.outer(mean, mean))
        return mean, cov
    # general (non-Gaussian) fallback: moment-weighted (less stable at high degree)
    moment_vec = moments_from_cumulants(kappa, 2 * p)
    mean = jet_expect(K1, moment_vec)
    E_cov = jet_expect(K2, moment_vec)
    E_mmT = jet_expect(jet_outer(K1, K1, 2 * p), moment_vec)
    cov = E_cov + (E_mmT - np.outer(mean, mean))
    return mean, cov


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def symbolic_kprop(layers, K1, K2, kappa, p, n_gl=64, var_floor=1e-12):
    """layers: list of ('linear', W, b) or ('relu',). K1/K2: input jets.

    kappa: hidden cumulants (index 0 unused, kappa[1]=0). Returns (mean, cov, diag).
    """
    moment_vec = moments_from_cumulants(kappa, p)   # for the tail diagnostic only
    diag = {"tail_scores": []}
    for layer in layers:
        if layer[0] == "linear":
            _, W, b = layer
            K1, K2 = linear_pushforward(K1, K2, W, b)
        elif layer[0] == "relu":
            K1, K2 = activation_relu(K1, K2, p, n_gl=n_gl, var_floor=var_floor)
            K1, K2 = jet_truncate(K1, p), jet_truncate(K2, p)
            diag["tail_scores"].append(tail_score(K1, moment_vec))
        else:
            raise ValueError(layer[0])
    mean, cov = marginalize(K1, K2, kappa)
    return mean, cov, diag


def make_input_state(d, p, V=None):
    """Input conditional cumulants of X|h for X~N(0,I).

    V: unit direction (d,) of the scalar latent h=V^T X (q=1). None -> q=0.
    q=1:  K1(h)=V dh (deg 1), K2 = I - V V^T (deg 0), h~N(0,1).
    q=0:  K1=0, K2=I.
    """
    K1 = jet_zeros(p, (d,))
    K2 = jet_zeros(p, (d, d))
    if V is None:
        K2[0] = np.eye(d)
        kappa = [0.0]                       # no hidden mode
    else:
        V = V / np.linalg.norm(V)
        K1[1] = V
        K2[0] = np.eye(d) - np.outer(V, V)
        kappa = gaussian_kappa(p_hidden=2, var=1.0)
    return K1, K2, kappa


# ===========================================================================
# Cross-checks used by the validation script
# ===========================================================================
def direct_k2_relu_kprop(layers, mu0, Sig0, n_gl=64, var_floor=1e-12):
    """Plain single-Gaussian k=2 ReLU cumulant propagation (NO hidden mode).

    The q=0 target: mean/cov tracked as plain tensors (degree-0 jets).
    """
    mu, Sig = mu0.copy(), Sig0.copy()
    for layer in layers:
        if layer[0] == "linear":
            _, W, b = layer
            mu = W @ mu + (b if b is not None else 0.0)
            Sig = W @ Sig @ W.T
        elif layer[0] == "relu":
            Sig = 0.5 * (Sig + Sig.T)
            var = np.maximum(np.diag(Sig), var_floor)
            r1 = relu_m1(mu, var, eps=var_floor)
            r2 = relu_pair_matrix(mu, Sig, n_gl=n_gl, eps=var_floor)
            mu = r1
            Sig = r2 - np.outer(r1, r1)
    return mu, Sig


def node_average_kprop(layers, d, V, n_nodes=41, n_gl=64):
    """skprop-style baseline: run the SAME k=2 closure at GH nodes of the
    scalar latent h and average (the 'old approach' the symbolic method replaces).
    """
    V = V / np.linalg.norm(V)
    t, om = hermgauss(n_nodes)
    nodes, w = t * SQRT2, om / math.sqrt(math.pi)
    Sig0 = np.eye(d) - np.outer(V, V)
    mean = np.zeros(layers_out_dim(layers, d))
    for h, wk in zip(nodes, w):
        mu0 = V * h
        m, _ = direct_k2_relu_kprop(layers, mu0, Sig0, n_gl=n_gl)
        mean += wk * m
    return mean


def layers_out_dim(layers, d):
    for layer in reversed(layers):
        if layer[0] == "linear":
            return layer[1].shape[0]
    return d


def mc_forward_mean(layers, d, n_samples=400_000, seed=0, batch=20000):
    """Monte-Carlo E[out] over X~N(0,I_d) through the ReLU net (ground truth)."""
    rng = np.random.default_rng(seed)
    out_dim = layers_out_dim(layers, d)
    acc = np.zeros(out_dim)
    n = 0
    while n < n_samples:
        b = min(batch, n_samples - n)
        x = rng.standard_normal((b, d))
        h = x
        for layer in layers:
            if layer[0] == "linear":
                _, W, bb = layer
                h = h @ W.T + (bb if bb is not None else 0.0)
            else:
                h = np.maximum(h, 0.0)
        acc += h.sum(0)
        n += b
    return acc / n

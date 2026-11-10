"""randomization.py: random formal contexts used as references for CLasIC.

Two families of references are provided (Chapter 5 of the thesis):

1. Bernoulli contexts and ensembles, which only fix size and density (and, for
   ensembles, how similar the contexts are). They describe the region of each entropy
   triangle that can be reached at all because of the closure of FCA.

   - ``bernoulli_context(n, m, p, rng)``: each incidence present with probability p.
   - ``bernoulli_ensemble(p, eps, n_ctx, shape, rng)``: a template context and n_ctx
     copies in which every incidence is redrawn with probability eps (eps = 0: identical
     copies, eps = 1: independent contexts). All copies keep density p.
   - ``ensemble_consistency(contexts)``: density p̂, mean pairwise disagreement D̄ and the
     redraw probability ε̂ that reproduces D̄, using
         D̄ = [1 - (1 - ε)^2] · 2p(1 - p)  =>  ε̂ = 1 - sqrt(1 - D̄ / (2 p̂ (1 - p̂))).

2. Margin-preserving randomizations of an empirical context, which keep the column
   sums (specificity of each neuron), the row sums (sparseness of each odorant) or both:

   - ``column_preserving(C, rng)``: permute the incidences of every column.
   - ``row_preserving(C, rng)``: permute the incidences of every row.
   - ``curveball(C, rng)``: uniform sample with both margins fixed (swap randomization,
     curveball algorithm of Strona et al. 2014), re-exported from fastclasic.

For large single contexts (e.g. dense 40 x 40), ``single_context_entropies`` uses exact
closed forms that avoid building the extent/intent matrices: in one context all concepts
are distinct and equally likely, so for a binary variable present in a fraction q of the
concepts ΔH' = 1 - h(q), VI' = 2t/N (t = pairs of concepts differing only in that
variable) and M' = h(q) - VI', and the CMET of each side is (1 - log2 N / n_var,
log2 N / n_var, 0). numba is used if installed.
"""

from __future__ import annotations

import math

import numpy as np

from fastclasic import curveball  # noqa: F401  (re-exported)

try:  # optional acceleration of the twin count
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap if not (args and callable(args[0])) else args[0]


# ------------------------------------------------------------------ Bernoulli

def density_levels(n=20, lo=0.01, mid=0.5, hi=0.99):
    """Densities geometrically spaced in [lo, mid) and (mid, hi] (20 levels by default)."""
    h = n // 2
    return np.concatenate([np.geomspace(lo, mid, h + 1)[:-1], np.geomspace(mid, hi, h + 1)[1:]])


def bernoulli_context(n, m, p, rng):
    return (rng.random((n, m)) < p).astype(np.uint8)


def bernoulli_ensemble(p, eps, n_ctx, shape, rng):
    T = rng.random(shape) < p
    return [np.where(rng.random(shape) < eps, rng.random(shape) < p, T).astype(np.uint8)
            for _ in range(n_ctx)]


def ensemble_consistency(contexts):
    """(p̂, D̄, ε̂) of a set of binary contexts of equal shape."""
    B = np.asarray(contexts).reshape(len(contexts), -1).astype(float)
    p = B.mean()
    k = B.shape[0]
    D = np.mean([np.mean(B[i] != B[j]) for i in range(k) for j in range(i + 1, k)])
    den = 2 * p * (1 - p)
    eps = 0.0 if den == 0 else float(1 - np.sqrt(max(0.0, 1 - min(1.0, D / den))))
    return float(p), float(D), eps


# ------------------------------------------------------------------ margins

def column_preserving(C, rng):
    C = np.asarray(C)
    return np.take_along_axis(C, np.argsort(rng.random(C.shape), axis=0), axis=0)


def row_preserving(C, rng):
    C = np.asarray(C)
    return np.take_along_axis(C, np.argsort(rng.random(C.shape), axis=1), axis=1)


MARGIN_NULLS = {
    'swap': curveball,                      # both margins
    'column_preserving': column_preserving,  # neuron specificity only
    'row_preserving': row_preserving,        # odorant sparseness only
}


# ------------------------------------------------------------------ closed forms

@njit(cache=True)
def _count_twins(srt, b):
    n = srt.shape[0]
    i = 0
    t = 0
    for k in range(n):
        x = srt[k]
        if x & b:
            y = x - b
            while i < n and srt[i] < y:
                i += 1
            if i < n and srt[i] == y:
                t += 1
    return t


def enumerate_codes(C):
    """Extent codes (bits = objects) and intent codes (bits = attributes) of all concepts."""
    C = np.asarray(C, dtype=np.uint8)
    n, m = C.shape
    w = np.uint64(1) << np.arange(n, dtype=np.uint64)
    cols = (C.astype(np.uint64) * w[:, None]).sum(axis=0).astype(np.uint64)
    S = np.array([np.uint64((1 << n) - 1)], dtype=np.uint64)
    for c in cols:
        S = np.unique(np.concatenate([S, S & c]))
    I = np.zeros(len(S), dtype=np.uint64)
    for j, c in enumerate(cols):
        I |= ((S & c) == S).astype(np.uint64) << np.uint64(j)
    return S, I


def _smet_closed(codes, nvar):
    N = len(codes)
    srt = np.sort(codes)
    out = np.empty((nvar, 3))
    for j in range(nvar):
        b = np.uint64(1) << np.uint64(j)
        q = float(((srt & b) != 0).mean())
        h = 0.0 if q in (0.0, 1.0) else -(q * math.log2(q) + (1 - q) * math.log2(1 - q))
        vi = 2.0 * _count_twins(srt, b) / N
        out[j] = (1.0 - h, h - vi, vi)
    return out


def single_context_entropies(C):
    """Exact SMET of intents and extents and CMET of one context (closed forms).

    Returns (smet_intents, smet_extents, cmet_X, cmet_Y): per-variable rows
    (ΔH', M', VI') and channel triplets (ΔH', I', H'cond).
    """
    C = np.asarray(C, dtype=np.uint8)
    n, m = C.shape
    S, I = enumerate_codes(C)
    N = len(S)
    hx = math.log2(N) if N > 1 else 0.0

    def vary(codes, k):
        return sum(1 for j in range(k) if 0 < int(((codes >> np.uint64(j)) & np.uint64(1)).sum()) < N)

    HUx, HUy = vary(S, n), vary(I, m)
    X = np.array([1 - hx / HUx, hx / HUx, 0.0]) if HUx else np.zeros(3)
    Y = np.array([1 - hx / HUy, hx / HUy, 0.0]) if HUy else np.zeros(3)
    return _smet_closed(I, m), _smet_closed(S, n), X, Y

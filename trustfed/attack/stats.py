"""Numeric helpers for the adaptive attacks.

Kept separate from :mod:`trustfed.attack.adaptive` so the attack strategies read
as strategies rather than as numerics. Nothing here is attack-specific: it is an
inverse normal CDF, the "A Little Is Enough" deviation coefficient derived from
it, and a distance matrix.
"""

from __future__ import annotations

import math

import numpy as np

from trustfed.attack.errors import AttackError


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Accurate to about 1.15e-9 in absolute value over ``(0, 1)``, which is far
    beyond what the attack needs. Implemented locally so the package keeps its
    numpy-only dependency footprint.
    """
    if not 0.0 < p < 1.0:
        raise AttackError(f"_norm_ppf requires p in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def pairwise_distances(U: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between the rows of ``U``."""
    diff = U[:, None, :] - U[None, :, :]
    return np.linalg.norm(diff, axis=2)


def alie_z(n_clients: int, n_malicious: int) -> float:
    """The "A Little Is Enough" deviation coefficient ``z_max``.

    With ``n`` clients and ``f`` attackers, an attacker needs
    ``s = floor(n/2 + 1) - f`` honest supporters on its side of the order
    statistic; the largest safe deviation is the ``(n - f - s)/(n - f)`` normal
    quantile.

    The degenerate branches do **not** return a negligible value: the result is
    ``1.0`` when ``f`` is outside ``(0, n)`` (there is no meaningful attacker
    fraction to solve for, so the caller gets one standard deviation of
    displacement) and ``0.5`` when the required supporter count exhausts the
    honest majority (half a standard deviation). Both are substantial
    displacements, chosen so a caller that ignores the degenerate case still
    produces a live attack rather than a no-op; they are conventions of this
    implementation, not results from the paper.
    """
    n, f = int(n_clients), int(n_malicious)
    if n <= 0 or f <= 0 or f >= n:
        return 1.0
    s = math.floor(n / 2.0 + 1.0) - f
    denom = n - f
    numer = denom - s
    if denom <= 0 or numer <= 0:
        return 0.5
    p = min(max(numer / denom, 1e-6), 1.0 - 1e-6)
    return abs(_norm_ppf(p))



__all__ = ["alie_z", "pairwise_distances"]

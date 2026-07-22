"""Robust aggregators for federated model updates.

All functions take a list of 1-D parameter vectors (one per surviving client)
and return a single aggregated vector of the same shape.

References
---------
* FedAvg: McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data", AISTATS 2017.
* Krum / Multi-Krum: Blanchard et al., "Machine Learning with Adversaries:
  Byzantine Tolerant Gradient Descent", NeurIPS 2017.
* Coordinate median / trimmed mean: Yin et al., "Byzantine-Robust Distributed
  Learning: Towards Optimal Statistical Rates", ICML 2018.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np


def _stack(updates: List[np.ndarray]) -> np.ndarray:
    if len(updates) == 0:
        raise ValueError("no updates to aggregate")
    return np.stack([np.asarray(u, dtype=float) for u in updates], axis=0)


def fedavg(updates, *, weights: Optional[np.ndarray] = None, **_) -> np.ndarray:
    """Weighted mean. Fast and accurate, but has zero Byzantine tolerance:
    a single large malicious update can dominate the average."""
    U = _stack(updates)
    if weights is None:
        return U.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return U.mean(axis=0)
    w = w / w.sum()
    return (U * w[:, None]).sum(axis=0)


def coordinate_median(updates, **_) -> np.ndarray:
    """Per-coordinate median. Tolerates < 50% Byzantine clients."""
    return np.median(_stack(updates), axis=0)


def trimmed_mean(updates, *, beta: float = 0.1, **_) -> np.ndarray:
    """Per-coordinate trimmed mean: drop the top/bottom ``beta`` fraction of
    values per coordinate before averaging."""
    U = np.sort(_stack(updates), axis=0)
    n = U.shape[0]
    k = int(np.floor(beta * n))
    if 2 * k >= n:
        k = max((n - 1) // 2, 0)
    if k == 0:
        return U.mean(axis=0)
    return U[k : n - k].mean(axis=0)


def _pairwise_sq_dists(U: np.ndarray) -> np.ndarray:
    sq = np.sum(U * U, axis=1)
    d = sq[:, None] + sq[None, :] - 2.0 * (U @ U.T)
    np.maximum(d, 0.0, out=d)
    return d


def krum(
    updates,
    *,
    n_byzantine: int = 1,
    multi: bool = False,
    m: Optional[int] = None,
    **_,
) -> np.ndarray:
    """(Multi-)Krum. Selects the update(s) closest to their nearest neighbors,
    which excludes far-away malicious updates.

    Requires ``n > 2f + 2`` where ``f = n_byzantine``; if that fails it falls
    back to the coordinate median.
    """
    U = _stack(updates)
    n = U.shape[0]
    f = int(n_byzantine)
    if n <= 2 * f + 2:
        return coordinate_median(updates)

    dists = _pairwise_sq_dists(U)
    num_neighbors = n - f - 2
    scores = np.empty(n, dtype=float)
    for i in range(n):
        ordered = np.sort(dists[i])  # ordered[0] is self-distance (0)
        scores[i] = float(np.sum(ordered[1 : num_neighbors + 1]))

    if multi:
        m = m if m is not None else (n - f)
        m = max(1, min(m, n))
        idx = np.argsort(scores)[:m]
        return U[idx].mean(axis=0)
    return U[int(np.argmin(scores))]


def multi_krum(updates, *, n_byzantine: int = 1, m: Optional[int] = None, **_):
    return krum(updates, n_byzantine=n_byzantine, multi=True, m=m)


AGGREGATORS: dict[str, Callable] = {
    "fedavg": fedavg,
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
}


def get_aggregator(name: str) -> Callable:
    try:
        return AGGREGATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown aggregator '{name}'. options: {sorted(AGGREGATORS)}"
        )

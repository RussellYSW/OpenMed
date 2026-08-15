"""Robust aggregators for federated model updates.

All functions take a list of 1-D parameter vectors (one per surviving client)
and return a single aggregated vector of the same shape. Every rule accepts and
ignores keyword arguments it does not use, so the server can call any of them
uniformly.

Each rule's docstring states its Byzantine tolerance; the same bounds are
available programmatically from :mod:`trustfed.aggregation.tolerance`, which is
what the benchmark harness reports.

References
---------
* FedAvg: McMahan et al., "Communication-Efficient Learning of Deep Networks
  from Decentralized Data", AISTATS 2017.
* Krum / Multi-Krum: Blanchard et al., "Machine Learning with Adversaries:
  Byzantine Tolerant Gradient Descent", NeurIPS 2017.
* Coordinate median / trimmed mean: Yin et al., "Byzantine-Robust Distributed
  Learning: Towards Optimal Statistical Rates", ICML 2018.
* Norm clipping / centered clipping: see :mod:`trustfed.aggregation.bounded`.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from trustfed.aggregation.bounded import centered_clipping, norm_clipped_mean
from trustfed.aggregation.core import (
    normalize_weights,
    pairwise_sq_dists,
    stack_updates,
)
from trustfed.aggregation.errors import AggregationError, UnknownAggregatorError


def _stack(updates: Sequence[np.ndarray]) -> np.ndarray:
    """Backwards-compatible alias for :func:`trustfed.aggregation.core.stack_updates`."""
    return stack_updates(updates)


def fedavg(
    updates: Sequence[np.ndarray],
    *,
    weights: Optional[Sequence[float]] = None,
    **_: object,
) -> np.ndarray:
    """Sample-count-weighted mean of the updates.

    Fast, unbiased and statistically efficient when every client is honest.

    Byzantine tolerance: **none** (f = 0). The mean has breakdown point 0 -- a
    single client submitting an unbounded vector moves the output arbitrarily
    far, and weighting by sample count makes it worse, since a client can lie
    about ``n_samples``. Present as the baseline that attacks must beat.

    Raises
    ------
    AggregationError
        On empty, ragged, non-finite input or invalid weights.
    """
    U = stack_updates(updates)
    w = normalize_weights(weights, U.shape[0])
    if w is None:
        return U.mean(axis=0)
    return (U * w[:, None]).sum(axis=0)


def coordinate_median(updates: Sequence[np.ndarray], **_: object) -> np.ndarray:
    """Per-coordinate median of the updates.

    Byzantine tolerance: ``f < n/2`` per coordinate (breakdown point 1/2). Does
    not protect against a colluding majority. Because each coordinate is taken
    independently the result need not be any client's actual update, which can
    hurt when honest sites are strongly non-IID.
    """
    return np.median(stack_updates(updates), axis=0)


def trimmed_mean(
    updates: Sequence[np.ndarray], *, beta: float = 0.1, **_: object
) -> np.ndarray:
    """Per-coordinate beta-trimmed mean.

    Sorts each coordinate across clients, drops the ``floor(beta * n)`` smallest
    and largest values, and averages the rest.

    Byzantine tolerance: ``f <= beta * n`` with ``2 * beta < 1``. The trim
    fraction must be set at least as large as the true attacker fraction --
    under-trimming leaves poison in the average, over-trimming silently deletes
    honest tail sites (see
    :func:`trustfed.aggregation.tolerance.recommended_trim_beta`).

    Raises
    ------
    AggregationError
        If ``beta`` is negative or not finite.
    """
    if not np.isfinite(beta) or beta < 0.0:
        raise AggregationError(f"beta must be a non-negative float, got {beta}")
    U = np.sort(stack_updates(updates), axis=0)
    n = U.shape[0]
    k = int(np.floor(beta * n))
    if 2 * k >= n:
        k = max((n - 1) // 2, 0)
    if k == 0:
        return U.mean(axis=0)
    return U[k : n - k].mean(axis=0)


def _krum_scores(U: np.ndarray, f: int) -> np.ndarray:
    """Krum score per client: sum of squared distances to its n-f-2 neighbours."""
    n = U.shape[0]
    dists = pairwise_sq_dists(U)
    num_neighbors = max(1, n - f - 2)
    scores = np.empty(n, dtype=float)
    for i in range(n):
        ordered = np.sort(dists[i])  # ordered[0] is the self-distance (0)
        scores[i] = float(np.sum(ordered[1 : num_neighbors + 1]))
    return scores


def krum(
    updates: Sequence[np.ndarray],
    *,
    n_byzantine: int = 1,
    multi: bool = False,
    m: Optional[int] = None,
    **_: object,
) -> np.ndarray:
    """(Multi-)Krum: keep the update(s) closest to their nearest neighbours.

    Byzantine tolerance: requires ``n > 2f + 2`` where ``f = n_byzantine``. When
    that precondition fails there is no valid Krum score, so this implementation
    **falls back to the coordinate median** rather than returning a value whose
    guarantee does not hold.

    Does not protect against colluders that submit mutually-close updates
    positioned just inside the honest cloud; that is exactly what the adaptive
    attack in :mod:`trustfed.attack.adaptive` does.

    Parameters
    ----------
    n_byzantine:
        The assumed number of Byzantine clients. Krum needs this estimate; if it
        is too small the selection can land on an attacker.
    multi:
        If ``True``, average the ``m`` best-scoring updates (Multi-Krum).
    m:
        Number of updates to average when ``multi``; defaults to ``n - f``.
    """
    U = stack_updates(updates)
    n = U.shape[0]
    f = int(n_byzantine)
    if f < 0:
        raise AggregationError(f"n_byzantine must be >= 0, got {n_byzantine}")
    if n <= 2 * f + 2:
        return coordinate_median(updates)

    scores = _krum_scores(U, f)
    if multi:
        m = m if m is not None else (n - f)
        m = max(1, min(int(m), n))
        idx = np.argsort(scores, kind="stable")[:m]
        return U[idx].mean(axis=0)
    return U[int(np.argmin(scores))]


def multi_krum(
    updates: Sequence[np.ndarray],
    *,
    n_byzantine: int = 1,
    m: Optional[int] = None,
    **_: object,
) -> np.ndarray:
    """Multi-Krum: mean of the ``m = n - f`` best-scoring updates.

    Byzantine tolerance: identical precondition to :func:`krum` (``n > 2f + 2``),
    with lower variance because it keeps more honest data per round.
    """
    return krum(updates, n_byzantine=n_byzantine, multi=True, m=m)


AGGREGATORS: Dict[str, Callable[..., np.ndarray]] = {
    "fedavg": fedavg,
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
    "norm_clip": norm_clipped_mean,
    "centered_clipping": centered_clipping,
}


def get_aggregator(name: str) -> Callable[..., np.ndarray]:
    """Look up an aggregation rule by registry name.

    Raises
    ------
    UnknownAggregatorError
        If ``name`` is not registered.
    """
    try:
        return AGGREGATORS[name]
    except KeyError:
        raise UnknownAggregatorError(
            f"unknown aggregator '{name}'. options: {sorted(AGGREGATORS)}"
        ) from None


def aggregator_name(fn: Callable[..., np.ndarray]) -> str:
    """Reverse lookup: registry name for an aggregator function.

    Returns the function's ``__name__`` if it is not a registered rule, so
    logging never fails on a user-supplied callable.
    """
    for key, value in AGGREGATORS.items():
        if value is fn:
            return key
    return getattr(fn, "__name__", "custom")


__all__ = [
    "AGGREGATORS",
    "get_aggregator",
    "aggregator_name",
    "fedavg",
    "coordinate_median",
    "trimmed_mean",
    "krum",
    "multi_krum",
    "norm_clipped_mean",
    "centered_clipping",
]

"""Shared array plumbing for the aggregation rules.

Kept in its own module so that :mod:`trustfed.aggregation.robust` and
:mod:`trustfed.aggregation.bounded` can share helpers without importing each
other.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from trustfed.aggregation.errors import AggregationError


def stack_updates(updates: Iterable[np.ndarray]) -> np.ndarray:
    """Stack per-client parameter vectors into an ``(n, d)`` matrix.

    Assumes every update is a 1-D vector of identical length and contains only
    finite values. Non-finite entries are rejected rather than silently
    propagated, because a single ``nan`` from one client would otherwise poison
    every coordinate of a mean-based rule.

    Raises
    ------
    AggregationError
        If the list is empty, shapes disagree, or a value is not finite.
    """
    items = [np.asarray(u, dtype=float) for u in updates]
    if not items:
        raise AggregationError("no updates to aggregate")
    shape = items[0].shape
    if len(shape) != 1:
        raise AggregationError(
            f"updates must be 1-D parameter vectors, got shape {shape}"
        )
    for i, u in enumerate(items):
        if u.shape != shape:
            raise AggregationError(
                f"update {i} has shape {u.shape}, expected {shape}"
            )
        if not np.all(np.isfinite(u)):
            raise AggregationError(
                f"update {i} contains non-finite values (nan/inf); "
                "reject it upstream instead of aggregating it"
            )
    return np.stack(items, axis=0)


def normalize_weights(
    weights: Optional[Sequence[float]], n: int
) -> Optional[np.ndarray]:
    """Return non-negative weights summing to 1, or ``None`` for uniform.

    Negative weights are an error: they would let one client subtract another's
    contribution. A zero-sum weight vector falls back to uniform.
    """
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise AggregationError(f"expected {n} weights, got shape {w.shape}")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise AggregationError("weights must be finite and non-negative")
    total = float(w.sum())
    if total <= 0.0:
        return None
    return w / total


def pairwise_sq_dists(U: np.ndarray) -> np.ndarray:
    """Symmetric matrix of squared Euclidean distances between rows of ``U``.

    Uses the expanded ``|a|^2 + |b|^2 - 2a.b`` form (fast), then clamps small
    negative values produced by floating-point cancellation.
    """
    sq = np.sum(U * U, axis=1)
    d = sq[:, None] + sq[None, :] - 2.0 * (U @ U.T)
    np.maximum(d, 0.0, out=d)
    np.fill_diagonal(d, 0.0)
    return d


def row_norms(U: np.ndarray) -> np.ndarray:
    """L2 norm of every row of ``U``."""
    return np.linalg.norm(U, axis=1)


__all__ = ["stack_updates", "normalize_weights", "pairwise_sq_dists", "row_norms"]

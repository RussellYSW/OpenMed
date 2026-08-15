"""Bounded-influence aggregation rules (norm clipping and centered clipping).

Median- and Krum-style rules defend by *selection*: they try to identify and
discard bad updates. The rules here defend by *limitation* instead: every client
is allowed to move the global model by at most a fixed distance, so a malicious
client's influence is bounded even when it is not detected. That is a weaker
guarantee (the attacker still moves the model a little) but a more graceful one:
it degrades smoothly as the number of attackers grows, and it does not need an
estimate of ``f`` to be correct.

References
----------
* Sun et al., "Can You Really Backdoor Federated Learning?", 2019 (norm bounding).
* Andrew et al., "Differentially Private Learning with Adaptive Clipping",
  NeurIPS 2021 (quantile-adaptive clipping radius).
* Karimireddy, He and Jaggi, "Learning from History for Byzantine Robust
  Optimization", ICML 2021 (centered clipping).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from trustfed.aggregation.core import normalize_weights, row_norms, stack_updates
from trustfed.aggregation.errors import AggregationError


def _reference_point(U: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
    """Return the clipping center: an explicit reference, else the coordinate median."""
    if reference is None:
        return np.median(U, axis=0)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (U.shape[1],):
        raise AggregationError(
            f"reference has shape {ref.shape}, expected {(U.shape[1],)}"
        )
    if not np.all(np.isfinite(ref)):
        raise AggregationError("reference contains non-finite values")
    return ref


def _clip_radius(
    deltas: np.ndarray, clip: Optional[float], quantile: float
) -> float:
    """Resolve the clipping radius, adaptively if ``clip`` is None."""
    if clip is not None:
        tau = float(clip)
        if not np.isfinite(tau) or tau <= 0.0:
            raise AggregationError(f"clip must be a positive finite float, got {clip}")
        return tau
    if not 0.0 < quantile <= 1.0:
        raise AggregationError(f"quantile must be in (0, 1], got {quantile}")
    norms = row_norms(deltas)
    tau = float(np.quantile(norms, quantile))
    # A degenerate round (all clients identical) gives tau == 0; fall back to a
    # tiny positive radius so the rule stays well defined.
    return tau if tau > 0.0 else 1e-12


def norm_clipped_mean(
    updates: Sequence[np.ndarray],
    *,
    weights: Optional[Sequence[float]] = None,
    clip: Optional[float] = None,
    reference: Optional[np.ndarray] = None,
    quantile: float = 0.5,
    **_: object,
) -> np.ndarray:
    """Clip every update into a ball around a reference point, then average.

    Each update ``x_i`` is replaced by ``ref + (x_i - ref) * min(1, tau/||x_i - ref||)``
    and the (optionally weighted) mean of the clipped updates is returned.

    Parameters
    ----------
    updates:
        Per-client parameter vectors.
    weights:
        Optional non-negative per-client weights (e.g. sample counts). Note that
        weighting *widens* the influence bound for a heavily weighted attacker.
    clip:
        Clipping radius ``tau``. If ``None``, it is set adaptively to the
        ``quantile`` of the observed deviation norms, which is itself a robust
        statistic for ``quantile <= 0.5``.
    reference:
        Center of the clipping ball. Defaults to the coordinate-wise median of
        the updates, so the center is itself robust. Passing the previous global
        model here gives the classic "bound how far one round can move" rule.
    quantile:
        Quantile used when ``clip is None``.

    Byzantine tolerance
    -------------------
    No exact-recovery threshold; instead a *bounded-influence* guarantee. With
    weights ``w_i`` summing to 1 and radius ``tau``, replacing the updates of a
    set ``B`` of clients shifts the output by at most ``2 * tau * sum_{i in B} w_i``
    (uniform weights: ``2 * tau * f / n``). The error therefore grows linearly in
    the attacker fraction rather than being unbounded (FedAvg) or zero-until-collapse
    (median). The reference point must itself be robust for this to be useful: with
    the default median reference the rule also inherits a ``f < n/2`` breakdown
    point for the center.

    Raises
    ------
    AggregationError
        On empty/ragged/non-finite input or an invalid radius.
    """
    U = stack_updates(updates)
    w = normalize_weights(weights, U.shape[0])
    ref = _reference_point(U, reference)
    deltas = U - ref[None, :]
    tau = _clip_radius(deltas, clip, quantile)

    norms = row_norms(deltas)
    scale = np.minimum(1.0, tau / np.maximum(norms, 1e-12))
    clipped = deltas * scale[:, None]
    if w is None:
        return ref + clipped.mean(axis=0)
    return ref + (clipped * w[:, None]).sum(axis=0)


def centered_clipping(
    updates: Sequence[np.ndarray],
    *,
    reference: Optional[np.ndarray] = None,
    tau: Optional[float] = None,
    iters: int = 3,
    quantile: float = 0.5,
    **_: object,
) -> np.ndarray:
    """Iterated centered clipping (Karimireddy et al., ICML 2021).

    Repeats ``v <- v + mean_i clip_tau(x_i - v)`` for ``iters`` rounds starting
    from ``reference``. Re-centering after each pass lets the estimate walk
    toward the honest cluster while never letting a single client contribute
    more than ``tau/n`` per pass.

    Parameters
    ----------
    updates:
        Per-client parameter vectors.
    reference:
        Starting center; defaults to the coordinate median. In a training loop
        the previous global model is the intended choice (that is what makes the
        rule "use history").
    tau:
        Clipping radius; adaptive (``quantile`` of deviation norms) if ``None``.
        The radius is recomputed at each pass when adaptive.
    iters:
        Number of clipping passes; must be >= 1. The original analysis uses a
        small constant (1-5).
    quantile:
        Quantile used for the adaptive radius.

    Byzantine tolerance
    -------------------
    Under the paper's assumptions (bounded honest-update variance and a
    reference within a bounded distance of the honest mean) centered clipping
    tolerates an attacker fraction ``delta = f/n < 1/2``, with error growing as
    ``O(sqrt(delta) * sigma)``. Unlike Krum it needs no explicit ``f``, but it
    does inherit the quality of the reference point: an attacker who has already
    dragged the reference far from the honest cluster keeps that advantage.

    Raises
    ------
    AggregationError
        On empty/ragged/non-finite input or ``iters < 1``.
    """
    if int(iters) < 1:
        raise AggregationError(f"iters must be >= 1, got {iters}")
    U = stack_updates(updates)
    v = _reference_point(U, reference)
    for _pass in range(int(iters)):
        deltas = U - v[None, :]
        radius = _clip_radius(deltas, tau, quantile)
        norms = row_norms(deltas)
        scale = np.minimum(1.0, radius / np.maximum(norms, 1e-12))
        v = v + (deltas * scale[:, None]).mean(axis=0)
    return v


__all__ = ["norm_clipped_mean", "centered_clipping"]

"""Byzantine-robust aggregation rules.

Every rule has the signature ``rule(updates, *, weights=None, n_byzantine=0,
**kwargs) -> np.ndarray`` and returns the aggregated parameter vector. Rules
ignore keyword arguments they do not need, so the server can call them
uniformly.

Two families are provided:

* **Selection rules** (median, trimmed mean, Krum, Multi-Krum) try to exclude
  malicious updates and give an exact-recovery guarantee below a threshold on
  the attacker fraction.
* **Bounded-influence rules** (norm clipping, centered clipping) cap how far any
  single client can move the model, degrading gracefully instead of collapsing.

:mod:`trustfed.aggregation.tolerance` records the assumed Byzantine-tolerance
bound of each rule and what it does *not* protect against.
"""

from __future__ import annotations

from trustfed.aggregation.bounded import centered_clipping, norm_clipped_mean
from trustfed.aggregation.errors import AggregationError, UnknownAggregatorError
from trustfed.aggregation.robust import (
    AGGREGATORS,
    aggregator_name,
    coordinate_median,
    fedavg,
    get_aggregator,
    krum,
    multi_krum,
    trimmed_mean,
)
from trustfed.aggregation.tolerance import (
    ToleranceBound,
    all_bounds,
    describe,
    bounded_influence_bound,
    max_byzantine,
    recommended_trim_beta,
    tolerance_bound,
    tolerance_table,
)

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
    "AggregationError",
    "UnknownAggregatorError",
    "ToleranceBound",
    "tolerance_bound",
    "bounded_influence_bound",
    "max_byzantine",
    "tolerance_table",
    "all_bounds",
    "recommended_trim_beta",
    "describe",
]

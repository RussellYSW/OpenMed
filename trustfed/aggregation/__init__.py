"""Byzantine-robust aggregation rules.

Every rule has the signature ``rule(updates, *, weights=None, n_byzantine=0,
**kwargs) -> np.ndarray`` and returns the aggregated parameter vector. Rules
ignore keyword arguments they do not need, so the server can call them
uniformly.
"""

from trustfed.aggregation.robust import (
    AGGREGATORS,
    coordinate_median,
    fedavg,
    get_aggregator,
    krum,
    trimmed_mean,
)

__all__ = [
    "AGGREGATORS",
    "get_aggregator",
    "fedavg",
    "coordinate_median",
    "trimmed_mean",
    "krum",
]

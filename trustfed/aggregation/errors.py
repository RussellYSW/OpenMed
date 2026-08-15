"""Typed exceptions raised by :mod:`trustfed.aggregation`."""

from __future__ import annotations


class AggregationError(ValueError):
    """Raised when updates cannot be aggregated or a rule is misconfigured.

    Subclasses :class:`ValueError` so existing callers that catch ``ValueError``
    keep working.
    """


class UnknownAggregatorError(AggregationError, KeyError):
    """Raised when an aggregator is requested by a name that is not registered."""

    def __str__(self) -> str:  # KeyError's repr would quote the message
        return self.args[0] if self.args else ""


__all__ = ["AggregationError", "UnknownAggregatorError"]

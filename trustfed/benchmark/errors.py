"""Typed exceptions raised by :mod:`trustfed.benchmark`."""

from __future__ import annotations


class BenchmarkError(ValueError):
    """Raised when a scenario is invalid or a benchmark run cannot be completed."""


__all__ = ["BenchmarkError"]

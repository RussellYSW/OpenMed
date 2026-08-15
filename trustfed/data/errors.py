"""Typed exceptions raised by :mod:`trustfed.data`."""

from __future__ import annotations


class DataError(ValueError):
    """Raised when a cohort specification is invalid or inconsistent.

    Subclasses :class:`ValueError` so that callers written against the stdlib
    contract keep working, while code that wants to catch *only* TrustFed data
    problems can catch :class:`DataError`.
    """


__all__ = ["DataError"]

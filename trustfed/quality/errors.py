"""Typed exceptions raised by :mod:`trustfed.quality`."""

from __future__ import annotations


class QualityError(ValueError):
    """Base class for quality-analysis failures."""


class BundleFormatError(QualityError):
    """Raised when a submitted bundle cannot be read at all.

    Individual *missing fields* are not errors -- they are findings, reported as
    check results. This exception is for a bundle that is not even inspectable
    (e.g. ``None``, or a scalar where an object was expected).
    """


class CheckConfigError(QualityError):
    """Raised when a check is constructed with invalid thresholds."""


__all__ = ["QualityError", "BundleFormatError", "CheckConfigError"]

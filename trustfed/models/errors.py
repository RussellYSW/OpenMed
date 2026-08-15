"""Typed exceptions raised by :mod:`trustfed.models`."""

from __future__ import annotations


class ModelError(ValueError):
    """Raised when model parameters or training data have the wrong shape/values."""


__all__ = ["ModelError"]

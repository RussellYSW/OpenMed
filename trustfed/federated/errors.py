"""Typed exceptions raised by :mod:`trustfed.federated`."""

from __future__ import annotations


class FederatedError(RuntimeError):
    """Base class for federated-training failures."""


class NoAcceptedUpdatesError(FederatedError):
    """Raised when every update in a round was rejected, leaving nothing to aggregate.

    This is deliberately fatal rather than silently skipped: continuing would
    mean publishing a round that no attested client contributed to.
    """


class SelectionError(FederatedError, ValueError):
    """Raised when a client-selection policy is misconfigured or has no candidates."""


class ClientConfigError(FederatedError, ValueError):
    """Raised when a client is constructed with inconsistent data or attack settings."""


__all__ = [
    "FederatedError",
    "NoAcceptedUpdatesError",
    "SelectionError",
    "ClientConfigError",
]

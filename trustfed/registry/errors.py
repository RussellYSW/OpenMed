"""Typed exceptions for the model registry."""

from __future__ import annotations


class RegistryError(Exception):
    """Base class for registry failures."""


class BundleNotFoundError(RegistryError, KeyError):
    """Raised when a bundle id is not present in the registry."""

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return RegistryError.__str__(self)


class DuplicateBundleError(RegistryError):
    """Raised when publishing content that is already in the registry.

    Bundle ids are content-addressed, so identical content always collides;
    change the version, the weights, or the parent set instead.
    """


class ValidationError(RegistryError):
    """Raised when a model card, evaluation report, or bundle field is invalid."""


class AttestationRequiredError(RegistryError):
    """Raised when a publish is refused because its attestation did not verify."""


class LineageError(RegistryError):
    """Raised when a lineage walk cannot complete (e.g. a cycle)."""


__all__ = [
    "AttestationRequiredError",
    "BundleNotFoundError",
    "DuplicateBundleError",
    "LineageError",
    "RegistryError",
    "ValidationError",
]

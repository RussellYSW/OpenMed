"""Typed exceptions for the attestation package."""

from __future__ import annotations


class AttestationError(Exception):
    """Base class for attestation failures (also raised when a quote is bad)."""


class PolicyError(AttestationError):
    """Raised when an attestation policy is malformed or self-contradictory."""


class NonceError(AttestationError):
    """Raised when a challenge nonce is unknown, expired, or already spent."""


__all__ = ["AttestationError", "NonceError", "PolicyError"]

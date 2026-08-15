"""Typed exceptions for the credit and reciprocity layer."""

from __future__ import annotations


class IncentiveError(Exception):
    """Base class for incentive-layer failures."""


class UnknownContributorError(IncentiveError, KeyError):
    """Raised when an actor has no recorded credit events."""

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return IncentiveError.__str__(self)


class UnknownCreditKindError(IncentiveError):
    """Raised when a credit event uses a kind the policy does not weight."""


class AttestationRejectedError(IncentiveError):
    """Raised when a counterparty attestation is missing, malformed or unverified."""


class IdentifierError(IncentiveError):
    """Raised when a citable identifier is malformed."""


class ReciprocityRefused(IncentiveError):
    """Raised when a refused reciprocity decision is escalated to an exception.

    Carries the machine-readable ``reason_code`` from the decision.
    """

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"[{reason_code}] {detail}")
        self.reason_code = reason_code
        self.detail = detail


__all__ = [
    "AttestationRejectedError",
    "IdentifierError",
    "IncentiveError",
    "ReciprocityRefused",
    "UnknownContributorError",
    "UnknownCreditKindError",
]

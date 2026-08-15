"""Typed exceptions for multi-party certification."""

from __future__ import annotations


class CertificationError(Exception):
    """Base class for certification failures."""


class UnknownReviewerError(CertificationError):
    """Raised when a reviewer id has no registered institution-bound key."""


class ConflictOfInterestError(CertificationError):
    """Raised when a reviewer is not permitted to review this model.

    Carries a machine-readable ``reason_code`` so callers can distinguish
    self-certification from other conflicts.
    """

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"[{reason_code}] {detail}")
        self.reason_code = reason_code
        self.detail = detail


class ThresholdNotMetError(CertificationError):
    """Raised when certification is requested before the policy is satisfied."""


class InvalidTransitionError(CertificationError):
    """Raised when a case is asked to move to a state the machine forbids."""


class UnknownCaseError(CertificationError, KeyError):
    """Raised when no certification case exists for a bundle id."""

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return CertificationError.__str__(self)


class ManualValidationError(CertificationError):
    """Raised when a fine-tuning manual does not satisfy the fixed schema."""


class SignatureRejectedError(CertificationError):
    """Raised when a review signature does not verify under the reviewer's key."""


class KeyringReadOnlyError(CertificationError):
    """Raised when a verify-only keyring is asked to produce a signature."""


__all__ = [
    "CertificationError",
    "ConflictOfInterestError",
    "InvalidTransitionError",
    "KeyringReadOnlyError",
    "ManualValidationError",
    "SignatureRejectedError",
    "ThresholdNotMetError",
    "UnknownCaseError",
    "UnknownReviewerError",
]

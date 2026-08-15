"""Typed exceptions for the append-only ledger.

Every failure path in :mod:`trustfed.ledger` raises one of these; the package
never raises a bare :class:`Exception`.
"""

from __future__ import annotations


class LedgerError(Exception):
    """Base class for every ledger failure."""


class ChainVerificationError(LedgerError):
    """Raised when a chain fails structural, hash, or signature verification."""


class SignatureError(LedgerError):
    """Raised when a block signature cannot be produced or does not verify."""


class BlockNotFoundError(LedgerError, KeyError):
    """Raised when a lookup by index or payload hash matches no block."""

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return LedgerError.__str__(self)


class LedgerStorageError(LedgerError):
    """Raised when the on-disk representation is missing or malformed."""


__all__ = [
    "LedgerError",
    "ChainVerificationError",
    "SignatureError",
    "BlockNotFoundError",
    "LedgerStorageError",
]

"""Verification records for the hash chain: issues, verdicts and checkpoints.

Kept separate from the backend so that callers (registry, certification,
incentives) can depend on the shapes of a verification result without importing
storage machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from trustfed.ledger.errors import ChainVerificationError, LedgerError


@dataclass(frozen=True)
class ChainIssue:
    """One problem found while verifying a chain.

    ``code`` is machine-readable and stable; ``detail`` is for humans.
    """

    code: str
    index: Optional[int]
    detail: str


@dataclass(frozen=True)
class Checkpoint:
    """An external anchor: how long the chain was and what its head hash was.

    Comparing a chain against a checkpoint recorded earlier is what turns a
    hash chain into truncation-evident storage: a chain with the tail cut off
    is internally consistent, but its length and head hash no longer match.
    """

    length: int
    head_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {"length": self.length, "head_hash": self.head_hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Checkpoint":
        """Rebuild from :meth:`to_dict` output."""
        try:
            return cls(int(data["length"]), str(data["head_hash"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerError(f"malformed checkpoint: {exc}") from exc


@dataclass(frozen=True)
class ChainVerification:
    """Structured verdict from :meth:`LedgerBackend.verify_chain`.

    ``anchored`` records whether the verification had an external
    :class:`Checkpoint` to compare length and head hash against. This matters
    because an unanchored verification cannot detect truncation: a chain with
    its tail removed is internally consistent. ``ok=True`` with
    ``anchored=False`` therefore means "nothing internally wrong with the
    blocks that are present", not "the chain is complete".
    """

    ok: bool
    n_blocks: int
    head_hash: str
    signatures_checked: bool
    issues: Tuple[ChainIssue, ...] = field(default_factory=tuple)
    anchored: bool = False

    @property
    def codes(self) -> Tuple[str, ...]:
        """Return the issue codes, in the order they were found."""
        return tuple(issue.code for issue in self.issues)

    def raise_if_invalid(self) -> None:
        """Raise :class:`ChainVerificationError` if verification failed."""
        if not self.ok:
            raise ChainVerificationError(
                "chain verification failed: "
                + "; ".join(f"[{i.code}@{i.index}] {i.detail}" for i in self.issues)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the verdict."""
        return {
            "ok": self.ok,
            "n_blocks": self.n_blocks,
            "head_hash": self.head_hash,
            "signatures_checked": self.signatures_checked,
            "anchored": self.anchored,
            "issues": [
                {"code": i.code, "index": i.index, "detail": i.detail}
                for i in self.issues
            ],
        }


__all__ = ["ChainIssue", "ChainVerification", "Checkpoint"]

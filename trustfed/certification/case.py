"""The per-bundle certification case record.

The authoritative history of a case is the append-only decision log; this
object is the current view derived from it, held by
:class:`~trustfed.certification.authority.CertificationAuthority`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from trustfed.certification.keys import ReviewSignature
from trustfed.certification.policy import SubmissionFacts
from trustfed.certification.states import CertificationState


@dataclass
class CertificationCase:
    """Mutable state of one bundle's certification.

    The authoritative history is the ledger; this object is the current view
    derived from it.

    ``verify`` is the predicate that decides whether a stored signature still
    counts. :class:`~trustfed.certification.authority.CertificationAuthority`
    sets it to its keyring's ``verify`` when it opens the case, so that
    :meth:`approving_institutions` and
    :meth:`~trustfed.certification.authority.CertificationAuthority.evaluate`
    answer the same question: without it, a reviewer whose key was revoked
    would stop counting towards the threshold while still appearing in the
    published snapshot. It is excluded from :meth:`to_dict` because it is a
    callable, not part of the record.
    """

    bundle_id: str
    facts: SubmissionFacts
    state: CertificationState = CertificationState.SUBMITTED
    signatures: List[ReviewSignature] = field(default_factory=list)
    submitted_by: str = ""
    manual_ok: Optional[bool] = None
    verify: Optional[Callable[[ReviewSignature], bool]] = field(
        default=None, repr=False, compare=False
    )

    def approving_institutions(self) -> Tuple[str, ...]:
        """Return the distinct institutions that currently approve.

        Signatures that no longer verify under ``verify`` (a revoked or rotated
        key) are dropped first, exactly as
        :meth:`~trustfed.certification.authority.CertificationAuthority.evaluate`
        drops them, and then the most recent signature per reviewer wins.

        This matches ``evaluate().institutions`` whenever the threshold outcome
        is about approvals. It does not match when a blocking review is on
        file: the outcome then reports no institutions at all, because a block
        halts the case regardless of who approved.
        """
        counted = [
            s for s in self.signatures if self.verify is None or self.verify(s)
        ]
        latest: Dict[str, ReviewSignature] = {s.reviewer_id: s for s in counted}
        return tuple(
            sorted({s.institution for s in latest.values() if s.decision == "approve"})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the case."""
        return {
            "bundle_id": self.bundle_id,
            "state": self.state.value,
            "facts": self.facts.to_dict(),
            "submitted_by": self.submitted_by,
            "manual_ok": self.manual_ok,
            "signatures": [s.to_dict() for s in self.signatures],
            "approving_institutions": list(self.approving_institutions()),
        }


__all__ = ["CertificationCase"]

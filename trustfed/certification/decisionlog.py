"""The append-only decision log, and the only way a case changes state.

Mixed into :class:`~trustfed.certification.authority.CertificationAuthority`.
Split out of that module so the writing side of the record can be read on its
own: every certification action -- submission, a signed review, a refusal, a
dispute, a revocation -- goes through :meth:`DecisionLogMixin._log` before
anything else happens, and every state change goes through
:meth:`DecisionLogMixin._transition`, which appends the decision *first* and
moves the case *second*.

That ordering is deliberate. If the append fails the case does not move; if the
case moves, the reason it moved is already in the chain. What the chain then
gives is tamper *evidence*: a decision cannot be edited or removed afterwards
without breaking the hash chain (see :mod:`trustfed.ledger`). It does not
authenticate who appended an entry -- that is the keyring's job, and the payload
records ``actor_authenticated`` so a reader can tell the difference.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from trustfed.certification.case import CertificationCase
from trustfed.certification.states import CertificationState, assert_transition
from trustfed.ledger.backend import ChainVerification
from trustfed.ledger.block import Block

#: Ledger payload ``event`` value for every certification action.
EVENT_CERTIFICATION = "certification_event"


class DecisionLogMixin:
    """Decision-log half of the certification authority.

    Assumes the host class provides ``_ledger`` and ``_clock``.
    """

    def _log(
        self,
        bundle_id: str,
        action: str,
        actor: str,
        *,
        from_state: Optional[CertificationState] = None,
        to_state: Optional[CertificationState] = None,
        detail: str = "",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Block:
        """Append one decision to the chain and return the block."""
        payload: Dict[str, Any] = {
            "event": EVENT_CERTIFICATION,
            "bundle_id": bundle_id,
            "action": action,
            "actor": actor,
            "from_state": from_state.value if from_state else None,
            "to_state": to_state.value if to_state else None,
            "detail": detail,
            "timestamp": self._clock(),
        }
        if extra:
            payload.update(dict(extra))
        return self._ledger.append(payload)

    def decision_log(self, bundle_id: Optional[str] = None) -> Tuple[Block, ...]:
        """Return the decision-log blocks, optionally filtered to one bundle."""
        return tuple(
            b
            for b in self._ledger
            if b.payload.get("event") == EVENT_CERTIFICATION
            and (bundle_id is None or b.payload.get("bundle_id") == bundle_id)
        )

    def verify_log(self) -> ChainVerification:
        """Verify the decision log's hash chain.

        A failure means someone edited or removed a recorded decision.
        """
        return self._ledger.verify_chain()

    def _transition(
        self,
        case: CertificationCase,
        target: CertificationState,
        action: str,
        actor: str,
        detail: str = "",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append the decision then move the case, refusing illegal moves."""
        assert_transition(case.state, target)
        self._log(
            case.bundle_id,
            action,
            actor,
            from_state=case.state,
            to_state=target,
            detail=detail,
            extra=extra,
        )
        case.state = target


__all__ = ["EVENT_CERTIFICATION", "DecisionLogMixin"]

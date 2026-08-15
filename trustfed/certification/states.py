"""The certification / dispute / appeal state machine.

States and the transitions between them are fixed here so that every actor --
authority, reviewer, disputing site -- moves a case through the same graph, and
so that an illegal move is a typed error rather than a silently-inconsistent
record.

::

    submitted ──review──▶ under_review ──threshold met──▶ certified
        │                     │  ▲                            │
        │                     │  └──re-review───┐             │
        └──dispute──▶ blocked ◀──dispute────────┼─────────────┘
                          │                     │
                          ├──remediation──▶ remediated
                          └──revoke──▶ revoked ◀──revoke──┘

``revoked`` is terminal: a revoked certification is never edited back into
existence, a new case must be opened for a new bundle.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Tuple

from trustfed.certification.errors import InvalidTransitionError


class CertificationState(str, Enum):
    """Lifecycle state of one certification case."""

    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    BLOCKED = "blocked"
    REMEDIATED = "remediated"
    CERTIFIED = "certified"
    REVOKED = "revoked"


#: Allowed state transitions, keyed by source state.
TRANSITIONS: Dict[CertificationState, FrozenSet[CertificationState]] = {
    CertificationState.SUBMITTED: frozenset(
        {
            CertificationState.UNDER_REVIEW,
            CertificationState.BLOCKED,
            CertificationState.REVOKED,
        }
    ),
    CertificationState.UNDER_REVIEW: frozenset(
        {
            CertificationState.CERTIFIED,
            CertificationState.BLOCKED,
            CertificationState.REVOKED,
        }
    ),
    CertificationState.BLOCKED: frozenset(
        {
            CertificationState.REMEDIATED,
            CertificationState.REVOKED,
        }
    ),
    CertificationState.REMEDIATED: frozenset(
        {
            CertificationState.UNDER_REVIEW,
            CertificationState.BLOCKED,
            CertificationState.REVOKED,
        }
    ),
    CertificationState.CERTIFIED: frozenset(
        {
            CertificationState.BLOCKED,
            CertificationState.REVOKED,
        }
    ),
    CertificationState.REVOKED: frozenset(),
}

#: States from which no further transition is possible.
TERMINAL_STATES: FrozenSet[CertificationState] = frozenset({CertificationState.REVOKED})


def can_transition(source: CertificationState, target: CertificationState) -> bool:
    """Return ``True`` iff ``source -> target`` is an allowed transition."""
    return target in TRANSITIONS.get(source, frozenset())


def assert_transition(
    source: CertificationState, target: CertificationState
) -> None:
    """Raise :class:`InvalidTransitionError` unless the transition is allowed."""
    if not can_transition(source, target):
        allowed = sorted(s.value for s in TRANSITIONS.get(source, frozenset()))
        raise InvalidTransitionError(
            f"cannot move a case from {source.value} to {target.value}; "
            f"allowed: {allowed or ['(terminal state)']}"
        )


def transitions_table() -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Return the transition table as plain strings, for docs and reports."""
    return tuple(
        (source.value, tuple(sorted(t.value for t in targets)))
        for source, targets in TRANSITIONS.items()
    )


__all__ = [
    "TERMINAL_STATES",
    "TRANSITIONS",
    "CertificationState",
    "assert_transition",
    "can_transition",
    "transitions_table",
]

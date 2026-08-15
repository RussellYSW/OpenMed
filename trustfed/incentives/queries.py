"""Reading the credit ledger: events, totals, standing and chain health.

Mixed into :class:`~trustfed.incentives.credit.CreditLedger`. Split from the
recording half so the two halves can be read separately: everything here is
derived from the append-only chain and never writes to it.

Every number produced here inherits the honesty caveat on the recording side --
the chain guarantees that recorded events have not been edited, not that the
work they describe was done, and not who appended them.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from trustfed.incentives.errors import UnknownContributorError
from trustfed.incentives.events import (
    EVENT_CREDIT,
    EVENT_RECIPROCITY,
    KIND_EVALUATION_SERVED,
    CreditEvent,
)
from trustfed.incentives.standing import ContributorStanding, StandingReport, tier_for
from trustfed.ledger.backend import ChainVerification


class CreditQueryMixin:
    """Read-only half of :class:`~trustfed.incentives.credit.CreditLedger`.

    Assumes the host class provides ``_ledger``, ``_tiers`` and ``_clock``.
    """

    # ---------------------------------------------------------------------- query

    def events(
        self, actor: Optional[str] = None, kind: Optional[str] = None
    ) -> Tuple[CreditEvent, ...]:
        """Return recorded credit events, optionally filtered by actor and kind."""
        out: List[CreditEvent] = []
        for block in self._ledger:
            if block.payload.get("event") != EVENT_CREDIT:
                continue
            event = CreditEvent.from_payload(block.payload)
            if actor is not None and event.actor != actor:
                continue
            if kind is not None and event.kind != kind:
                continue
            out.append(event)
        return tuple(out)

    def decisions(self) -> Tuple[Mapping[str, Any], ...]:
        """Return recorded reciprocity decisions as raw payloads."""
        return tuple(
            dict(b.payload)
            for b in self._ledger
            if b.payload.get("event") == EVENT_RECIPROCITY
        )

    def contributors(self) -> Tuple[str, ...]:
        """Return every actor with at least one credit event, first-seen order."""
        seen: List[str] = []
        for event in self.events():
            if event.actor not in seen:
                seen.append(event.actor)
        return tuple(seen)

    def credit(self, actor: str) -> float:
        """Return an actor's total weighted credit (0.0 if they have none)."""
        return float(sum(e.credit for e in self.events(actor)))

    def count(self, actor: str, kind: str) -> int:
        """Return how many events of ``kind`` an actor has recorded."""
        return sum(1 for _ in self.events(actor, kind))

    def served_evaluations(self, actor: str, *, attested_only: bool = False) -> int:
        """Return how many times ``actor`` has served as an evaluator.

        By default this counts every recorded claim, including ones the actor
        appended about itself. Pass ``attested_only=True`` to count only events
        carrying a counterparty attestation that verified when it was recorded
        (see :meth:`~trustfed.incentives.credit.CreditLedger.record_evaluation_served`).

        In ``attested_only`` mode the count is over *distinct* attestations --
        keyed by ``(counterparty, ref, attestation_signature)`` -- not over
        events. The recording path already refuses a resubmitted attestation;
        counting distinct signatures here means that even a chain written by
        some other tool, or one carrying events from an older version, cannot
        turn one signed acknowledgement into several units of standing. Attested
        events with no recorded signature (written before signatures were
        stored) are counted individually, since there is nothing to compare.
        """
        events = self.events(actor, KIND_EVALUATION_SERVED)
        if not attested_only:
            return len(events)
        seen: set = set()
        total = 0
        for event in events:
            if not event.attested:
                continue
            if not event.attestation_signature:
                total += 1
                continue
            key = (event.counterparty, event.ref, event.attestation_signature)
            if key in seen:
                continue
            seen.add(key)
            total += 1
        return total

    def __len__(self) -> int:
        """Return the number of credit events recorded."""
        return len(self.events())

    def __iter__(self) -> Iterator[CreditEvent]:
        """Iterate credit events in recorded order."""
        return iter(self.events())

    # ------------------------------------------------------------------- standing

    def standing(self, actor: str) -> ContributorStanding:
        """Return one contributor's standing.

        Raises
        ------
        UnknownContributorError
            If the actor has no recorded events.
        """
        events = self.events(actor)
        if not events:
            raise UnknownContributorError(f"no credit events for {actor!r}")
        counts: Dict[str, int] = {}
        institution = ""
        for event in events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
            institution = event.institution or institution
        total = float(sum(e.credit for e in events))
        return ContributorStanding(
            actor=actor,
            institution=institution,
            credit=total,
            tier=tier_for(total, self._tiers),
            event_counts=counts,
        )

    def standing_report(self) -> StandingReport:
        """Return a standing snapshot for every contributor.

        Rows are sorted by credit descending, then by actor id. The report
        records whether the underlying chain still verifies, so a reader can
        tell whether the numbers rest on an intact log.
        """
        rows = [self.standing(actor) for actor in self.contributors()]
        rows.sort(key=lambda r: (-r.credit, r.actor))
        return StandingReport(
            rows=tuple(rows),
            generated_at=str(self._clock()),
            ledger_verified=self.verify().ok,
            n_events=len(self.events()),
        )

    def verify(self) -> ChainVerification:
        """Verify the underlying hash chain backing these credit records."""
        return self._ledger.verify_chain()


__all__ = ["CreditQueryMixin"]

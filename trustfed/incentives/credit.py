"""The credit ledger: what each site contributed, recorded append-only.

Credit events sit on top of :mod:`trustfed.ledger`, so the contribution record
inherits the same tamper evidence as the lineage and certification records: the
chain detects retroactive edits to recorded credit; it does not authenticate who
appended an entry, so a site can append credit to itself. Appending is the
supported operation, and nothing here checks that the work described actually
happened. Read the ledger as "this is what was claimed, in this order, and it
has not been edited since", not as "this is what was done".

That gap matters most for ``evaluation_served``, because
:class:`~trustfed.incentives.reciprocity.ReciprocityPolicy` reads that count.
:meth:`CreditLedger.record_evaluation_served` is the tighter path: it requires
the evaluated site to name itself as counterparty and, optionally, to hand over
a signed attestation that is verified before the event is appended.

Credit *weights* are a governance parameter, not a measurement. The defaults
here express one plausible ordering (contributing data or serving an evaluation
costs a site more than running a training round) and are meant to be replaced by
whatever the deployment's governance body agrees.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from trustfed.incentives.attest import CounterpartyRegistry, ServiceAttestation
from trustfed.incentives.errors import (
    AttestationRejectedError,
    UnknownCreditKindError,
)
from trustfed.incentives.events import (
    DEFAULT_WEIGHTS,
    EVENT_CREDIT,
    EVENT_RECIPROCITY,
    KIND_EVALUATION_SERVED,
    CreditEvent,
)
from trustfed.incentives.queries import CreditQueryMixin
from trustfed.incentives.standing import DEFAULT_TIERS, StandingTier
from trustfed.ledger.backend import LedgerBackend
from trustfed.ledger.block import utc_now_iso
from trustfed.ledger.memory import InMemoryLedger

if TYPE_CHECKING:  # pragma: no cover - avoids a runtime import cycle
    from trustfed.incentives.identifiers import ReleaseAttribution
    from trustfed.incentives.reciprocity import ReciprocityDecision


class CreditLedger(CreditQueryMixin):
    """Append-only record of contributions, standings and reciprocity decisions.

    Parameters
    ----------
    ledger:
        Underlying hash chain. Defaults to a non-durable
        :class:`~trustfed.ledger.memory.InMemoryLedger`.
    weights:
        Credit weight per event kind. Recording a kind that is not weighted
        raises :class:`UnknownCreditKindError`, so typos do not silently earn
        zero credit.
    tiers:
        Standing ladder used by :meth:`standing`.
    clock:
        ISO-8601 timestamp source; injectable for deterministic tests.
    counterparties:
        Verifying keys for sites that may sign a
        :class:`~trustfed.incentives.attest.ServiceAttestation`. Defaults to an
        empty registry, in which case no attestation can verify.
    require_attestation:
        Refuse :meth:`record_evaluation_served` without a verified counterparty
        attestation. Default False, because the demo and tests drive the ledger
        without a key registry; set it True in a deployment where the
        reciprocity count must not be self-asserted.
    """

    def __init__(
        self,
        ledger: Optional[LedgerBackend] = None,
        *,
        weights: Optional[Mapping[str, float]] = None,
        tiers: Sequence[StandingTier] = DEFAULT_TIERS,
        clock: Optional[Callable[[], str]] = None,
        counterparties: Optional[CounterpartyRegistry] = None,
        require_attestation: bool = False,
    ) -> None:
        self._ledger: LedgerBackend = ledger if ledger is not None else InMemoryLedger()
        self._weights: Dict[str, float] = dict(
            weights if weights is not None else DEFAULT_WEIGHTS
        )
        self._tiers = tuple(tiers)
        self._clock = clock if clock is not None else utc_now_iso
        self._institutions: Dict[str, str] = {}
        self._counterparties = (
            counterparties if counterparties is not None else CounterpartyRegistry()
        )
        self._require_attestation = require_attestation

    # ------------------------------------------------------------------ properties

    @property
    def ledger(self) -> LedgerBackend:
        """The underlying append-only chain."""
        return self._ledger

    @property
    def counterparties(self) -> CounterpartyRegistry:
        """Registry of keys that may vouch for a service performed."""
        return self._counterparties

    @property
    def weights(self) -> Mapping[str, float]:
        """The credit weight per kind currently in force."""
        return dict(self._weights)

    @property
    def tiers(self) -> Tuple[StandingTier, ...]:
        """The standing ladder in force."""
        return self._tiers

    # --------------------------------------------------------------------- record

    def record(
        self,
        actor: str,
        kind: str,
        *,
        quantity: float = 1.0,
        institution: str = "",
        ref: str = "",
        detail: str = "",
        counterparty: str = "",
        attested: bool = False,
        attestation_signature: str = "",
        attestation_key_id: str = "",
    ) -> CreditEvent:
        """Append one contribution event and return it.

        SECURITY NOTE: this is unauthenticated and open to any caller. The
        ``actor`` is whatever the caller passes, so a site can append credit to
        itself, including ``evaluation_served`` credit that
        :class:`~trustfed.incentives.reciprocity.ReciprocityPolicy` then reads.
        The hash chain does not object, because appending is the supported
        operation; what it catches is a later edit to an entry already
        recorded. Use :meth:`record_evaluation_served` for the kind where a
        second party can vouch.

        Raises
        ------
        UnknownCreditKindError
            If ``kind`` has no configured weight.
        ValueError
            If ``quantity`` is not positive.
        """
        if kind not in self._weights:
            raise UnknownCreditKindError(
                f"unweighted credit kind {kind!r}; known kinds: "
                f"{sorted(self._weights)}"
            )
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        weight = float(self._weights[kind])
        event = CreditEvent(
            actor=actor,
            kind=kind,
            quantity=float(quantity),
            weight=weight,
            credit=weight * float(quantity),
            institution=institution or self._institutions.get(actor, ""),
            ref=ref,
            detail=detail,
            timestamp=str(self._clock()),
            counterparty=counterparty,
            attested=bool(attested),
            attestation_signature=attestation_signature,
            attestation_key_id=attestation_key_id,
        )
        if institution:
            self._institutions[actor] = institution
        payload = {"event": EVENT_CREDIT}
        payload.update(event.to_dict())
        self._ledger.append(payload)
        return event

    def record_evaluation_served(
        self,
        actor: str,
        *,
        counterparty: str,
        attestation: Optional["ServiceAttestation"] = None,
        quantity: float = 1.0,
        institution: str = "",
        ref: str = "",
        detail: str = "",
    ) -> CreditEvent:
        """Append ``evaluation_served`` credit naming the site that was served.

        ``counterparty`` is the evaluated site: the second party who can
        contradict the claim. It is required and must differ from ``actor``,
        so this credit can never be recorded as a purely internal event.

        Pass ``attestation`` -- a
        :class:`~trustfed.incentives.attest.ServiceAttestation` signed by the
        counterparty -- to have the claim verified against that site's
        registered key before it is appended. The resulting event carries
        ``attested=True``, and
        :meth:`served_evaluations` can be asked to count only those, which is
        what makes :class:`~trustfed.incentives.reciprocity.ReciprocityPolicy`
        a check on a second party's word rather than on self-assertion.

        Without an attestation the event is still recorded (unless this ledger
        was built with ``require_attestation=True``) but marked
        ``attested=False``, so a reader can tell the two apart.

        Raises
        ------
        AttestationRejectedError
            If ``counterparty`` is empty or equal to ``actor``; if an
            attestation is required and absent; if the supplied attestation
            does not match this event, is unsigned, or does not verify under
            the counterparty's registered key; or if that exact attestation has
            already been recorded on this chain. The last case is the replay
            guard: an attestation verifies every time it is presented, so
            without it one acknowledgement would buy an unlimited attested
            count. A second evaluation needs a second signed attestation.
        """
        if not counterparty:
            raise AttestationRejectedError(
                "evaluation_served credit needs the evaluated site as counterparty"
            )
        if counterparty == actor:
            raise AttestationRejectedError(
                f"{actor!r} cannot be its own counterparty for an evaluation"
            )
        if attestation is None:
            if self._require_attestation:
                raise AttestationRejectedError(
                    f"this credit ledger requires a signed attestation from "
                    f"{counterparty!r} before crediting {actor!r}"
                )
            attested = False
        else:
            self._check_attestation(attestation, actor, counterparty, ref)
            attested = True
        return self.record(
            actor,
            KIND_EVALUATION_SERVED,
            quantity=quantity,
            institution=institution,
            ref=ref,
            detail=detail,
            counterparty=counterparty,
            attested=attested,
            attestation_signature=attestation.signature if attestation else "",
            attestation_key_id=attestation.key_id if attestation else "",
        )

    def _check_attestation(
        self,
        attestation: "ServiceAttestation",
        actor: str,
        counterparty: str,
        ref: str,
    ) -> None:
        """Raise unless ``attestation`` covers this event and verifies."""
        if (
            attestation.actor != actor
            or attestation.counterparty != counterparty
            or attestation.kind != KIND_EVALUATION_SERVED
        ):
            raise AttestationRejectedError(
                "attestation does not cover this event: it credits "
                f"{attestation.actor!r} for {attestation.kind!r} on behalf of "
                f"{attestation.counterparty!r}"
            )
        if ref and attestation.ref != ref:
            raise AttestationRejectedError(
                f"attestation covers {attestation.ref!r}, not {ref!r}"
            )
        if not attestation.signature:
            raise AttestationRejectedError(
                "attestation carries no signature, so nothing was vouched for"
            )
        if not self._counterparties.verify(attestation):
            raise AttestationRejectedError(
                f"attestation from {counterparty!r} did not verify under a "
                "registered key"
            )
        self._reject_replay(attestation)

    def _reject_replay(self, attestation: "ServiceAttestation") -> None:
        """Raise if this exact attestation was already spent on this chain.

        A :class:`~trustfed.incentives.attest.ServiceAttestation` is a bearer
        token once it has been presented: it verifies every time. Without this
        check a single acknowledgement could be resubmitted to manufacture an
        unbounded attested-service count, which is precisely the count
        :class:`~trustfed.incentives.reciprocity.ReciprocityPolicy` reads when
        it is built with ``require_attested_service=True``.

        The scan is over the chain rather than over in-memory state, so it
        still holds for a ledger reopened over a persisted file, and it cannot
        be defeated by dropping the process. It is linear in the number of
        recorded events; this reference implementation targets thousands.

        A counterparty who genuinely wants to vouch for a second evaluation
        signs a second attestation -- ``signed_at`` differs, so the signature
        does too.
        """
        if not attestation.signature:  # pragma: no cover - refused earlier
            return
        for block in self._ledger:
            payload = block.payload
            if payload.get("event") != EVENT_CREDIT:
                continue
            if payload.get("attestation_signature") != attestation.signature:
                continue
            raise AttestationRejectedError(
                f"this attestation from {attestation.counterparty!r} was "
                f"already recorded (block {block.index}); a second evaluation "
                "needs a second signed attestation"
            )

    def record_release(
        self,
        attribution: "ReleaseAttribution",
        *,
        quantity: float = 1.0,
    ) -> Tuple[CreditEvent, ...]:
        """Record ``release_published`` credit for each attributed contributor.

        ``attribution`` is a
        :class:`~trustfed.incentives.identifiers.ReleaseAttribution`; its
        citable identifier becomes the ``ref`` on every event, so credit is
        traceable back to the release it came from.
        """
        events: List[CreditEvent] = []
        for contributor in attribution.contributors:
            events.append(
                self.record(
                    contributor.actor,
                    "release_published",
                    quantity=quantity,
                    institution=contributor.institution,
                    ref=attribution.identifier.uri,
                    detail=f"{contributor.role} on {attribution.title}",
                )
            )
        return tuple(events)

    def record_decision(self, decision: "ReciprocityDecision") -> None:
        """Append a reciprocity decision to the ledger, granted or refused."""
        payload = {"event": EVENT_RECIPROCITY, "timestamp": str(self._clock())}
        payload.update(decision.to_dict())
        self._ledger.append(payload)


__all__ = [
    "DEFAULT_WEIGHTS",
    "EVENT_CREDIT",
    "EVENT_RECIPROCITY",
    "KIND_EVALUATION_SERVED",
    "CreditEvent",
    "CreditLedger",
]

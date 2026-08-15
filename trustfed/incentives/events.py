"""Credit event records and the governance defaults that weight them.

Split out of :mod:`trustfed.incentives.credit` so the shape of a contribution
record can be imported (and the default weights inspected) without pulling in
the ledger machinery.

The weights below are a governance *parameter*, not a measurement: they express
one plausible ordering of effort and are meant to be replaced by whatever a
deployment's governance body agrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


#: Ledger payload ``event`` value for a credit entry.
EVENT_CREDIT = "credit_entry"
#: Ledger payload ``event`` value for a recorded reciprocity decision.
EVENT_RECIPROCITY = "reciprocity_decision"

#: Credit kinds and their default weights (governance defaults, not measurements).
DEFAULT_WEIGHTS: Mapping[str, float] = {
    "data_contribution": 3.0,
    "training_round": 1.0,
    "evaluation_served": 2.0,
    "review_signed": 2.0,
    "release_published": 3.0,
    "benchmark_hosted": 2.0,
    "maintenance": 1.0,
}

#: The kind counted by the default reciprocity rule.
KIND_EVALUATION_SERVED = "evaluation_served"


@dataclass(frozen=True)
class CreditEvent:
    """One recorded contribution.

    ``attestation_signature`` and ``attestation_key_id`` carry the identifying
    fields of the counterparty attestation that backed this event, when there
    was one. They are recorded so that a later submission of the *same*
    attestation is detectable from the chain alone: without them, one signed
    acknowledgement could be replayed to manufacture an unbounded attested
    count. They are not secrets -- a signature is public once it is presented.
    """

    actor: str
    kind: str
    quantity: float
    weight: float
    credit: float
    institution: str = ""
    ref: str = ""
    detail: str = ""
    timestamp: str = ""
    counterparty: str = ""
    attested: bool = False
    attestation_signature: str = ""
    attestation_key_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "actor": self.actor,
            "kind": self.kind,
            "quantity": self.quantity,
            "weight": self.weight,
            "credit": self.credit,
            "institution": self.institution,
            "ref": self.ref,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "counterparty": self.counterparty,
            "attested": self.attested,
            "attestation_signature": self.attestation_signature,
            "attestation_key_id": self.attestation_key_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CreditEvent":
        """Rebuild an event from a ledger payload."""
        return cls(
            actor=str(payload["actor"]),
            kind=str(payload["kind"]),
            quantity=float(payload.get("quantity", 1.0)),
            weight=float(payload.get("weight", 0.0)),
            credit=float(payload.get("credit", 0.0)),
            institution=str(payload.get("institution", "")),
            ref=str(payload.get("ref", "")),
            detail=str(payload.get("detail", "")),
            timestamp=str(payload.get("timestamp", "")),
            counterparty=str(payload.get("counterparty", "")),
            attested=bool(payload.get("attested", False)),
            attestation_signature=str(payload.get("attestation_signature", "")),
            attestation_key_id=str(payload.get("attestation_key_id", "")),
        )


__all__ = [
    "DEFAULT_WEIGHTS",
    "EVENT_CREDIT",
    "EVENT_RECIPROCITY",
    "KIND_EVALUATION_SERVED",
    "CreditEvent",
]

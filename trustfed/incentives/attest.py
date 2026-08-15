"""Counterparty attestations for credit that describes a service to someone else.

Most credit kinds are a site's own report of its own work: a training round, a
maintenance task. Nothing outside the site can confirm those, and the ledger
records them as claims. ``evaluation_served`` is different -- there is a second
party, the site that was evaluated -- and it is the kind the reciprocity rule
reads, so it is worth binding to that second party's word.

A :class:`ServiceAttestation` is that word: a short signed statement, produced by
the *counterparty* (the evaluated site), naming who served, who was served, what
was done and against which artefact.
:meth:`~trustfed.incentives.credit.CreditLedger.record_evaluation_served`
verifies it before appending the credit event.

SECURITY NOTE: this proves that the holder of the counterparty's registered key
signed the statement. It does not prove an evaluation was performed, and the key
registry here (:class:`CounterpartyRegistry`) is local and unauthenticated, in
the same way as :mod:`trustfed.certification.keys`. Two colluding sites can
still sign each other's statements; what this removes is a *single* site
inflating its own count unilaterally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from trustfed.incentives.errors import AttestationRejectedError
from trustfed.ledger.block import utc_now_iso
from trustfed.ledger.crypto import Signer, Verifier, canonical_json


@dataclass(frozen=True)
class ServiceAttestation:
    """A counterparty's signed acknowledgement that a service was performed.

    Attributes
    ----------
    actor:
        The site being credited -- the one that served.
    counterparty:
        The site that received the service and is signing this statement.
    kind:
        The credit kind being attested, e.g. ``"evaluation_served"``.
    ref:
        The artefact the service was performed against, typically a bundle id.
    """

    actor: str
    counterparty: str
    kind: str
    ref: str
    signed_at: str
    signature: str = ""
    key_id: str = ""

    def signing_material(self) -> bytes:
        """Return the exact bytes this signature covers."""
        return canonical_json(
            {
                "actor": self.actor,
                "counterparty": self.counterparty,
                "kind": self.kind,
                "ref": self.ref,
                "signed_at": self.signed_at,
            }
        )

    @classmethod
    def create(
        cls,
        signer: Signer,
        *,
        actor: str,
        counterparty: str,
        kind: str,
        ref: str,
        signed_at: Optional[str] = None,
    ) -> "ServiceAttestation":
        """Build and sign an attestation with the counterparty's key."""
        if actor == counterparty:
            raise AttestationRejectedError(
                "a site cannot be its own counterparty; the point of the "
                "attestation is that a second party vouched for the service"
            )
        unsigned = cls(
            actor=actor,
            counterparty=counterparty,
            kind=kind,
            ref=ref,
            signed_at=signed_at if signed_at is not None else utc_now_iso(),
        )
        return cls(
            actor=unsigned.actor,
            counterparty=unsigned.counterparty,
            kind=unsigned.kind,
            ref=unsigned.ref,
            signed_at=unsigned.signed_at,
            signature=signer.sign(unsigned.signing_material()),
            key_id=signer.key_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "actor": self.actor,
            "counterparty": self.counterparty,
            "kind": self.kind,
            "ref": self.ref,
            "signed_at": self.signed_at,
            "signature": self.signature,
            "key_id": self.key_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ServiceAttestation":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            actor=str(data["actor"]),
            counterparty=str(data["counterparty"]),
            kind=str(data["kind"]),
            ref=str(data.get("ref", "")),
            signed_at=str(data.get("signed_at", "")),
            signature=str(data.get("signature", "")),
            key_id=str(data.get("key_id", "")),
        )


class CounterpartyRegistry:
    """Verifying keys for the sites that may sign :class:`ServiceAttestation`.

    Registration is local and unauthenticated, exactly like
    :class:`trustfed.certification.keys.ReviewerKeyring`: whoever controls
    registration controls which sites can vouch for anything. It models the
    policy layer, not a PKI.
    """

    def __init__(self) -> None:
        self._verifiers: Dict[str, Verifier] = {}

    def register(self, actor: str, verifier: Verifier) -> None:
        """Bind ``actor`` to a verifying key. Re-registering rotates the key."""
        self._verifiers[actor] = verifier

    def __contains__(self, actor: object) -> bool:
        """Return ``True`` iff a verifying key is registered for ``actor``."""
        return actor in self._verifiers

    def __len__(self) -> int:
        """Return the number of registered counterparties."""
        return len(self._verifiers)

    def verify(self, attestation: ServiceAttestation) -> bool:
        """Return ``True`` iff the attestation verifies under a bound key.

        Returns ``False`` -- never raises -- when the counterparty is unknown,
        the recorded key id does not match the registered one, or the signature
        itself does not check out.
        """
        verifier = self._verifiers.get(attestation.counterparty)
        if verifier is None:
            return False
        if attestation.key_id and attestation.key_id != verifier.key_id:
            return False
        return verifier.verify(attestation.signing_material(), attestation.signature)


__all__ = ["CounterpartyRegistry", "ServiceAttestation"]

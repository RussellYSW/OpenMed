"""Signed authorisation for a dispute-side action on a certification case.

A :class:`~trustfed.certification.keys.ReviewSignature` says *what a reviewer
thinks of a model*. It deliberately does not say *what should be done to the
case*, and it is published in the decision log the moment it is filed. Using one
to authorise a state change is therefore unsound: anyone who can read the log
holds a signature that verifies, and a reviewer who filed an honest blocking
review would find it re-presented as authority for a terminal ``revoke`` they
never asked for.

:class:`ActionAuthorization` is the separate record for that job. It is signed
over the action itself, so a ``dispute`` authorisation cannot be spent as a
``revoke``, and it carries a single-use ``nonce`` so a given authorisation can
be spent exactly once --
:meth:`~trustfed.certification.disputes.DisputeActionsMixin._authenticate_actor`
refuses one it has already recorded, and refuses one whose signature already
appears in the decision log.

SCOPE NOTE, plainly: the nonce makes an authorisation *single-use*, not
*recent*. Nothing here proves when the signer decided; a signer can mint a
hundred authorisations in advance and hand them out. What this removes is the
replay of a signature that was produced for some other purpose, and the reuse of
one that has already been spent. Binding an authorisation to a challenge the
authority issued would additionally prove recency; that would need the authority
to hold outstanding-challenge state, which this reference implementation does
not model.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from trustfed.certification.errors import CertificationError
from trustfed.ledger.block import utc_now_iso
from trustfed.ledger.crypto import Signer, canonical_json

#: Dispute-side actions an :class:`ActionAuthorization` may authorise.
AUTHORIZED_ACTIONS: Tuple[str, ...] = (
    "dispute",
    "remediate",
    "appeal",
    "appeal_refused",
    "revoke",
)


@dataclass(frozen=True)
class ActionAuthorization:
    """One actor's signed instruction to perform one action on one case.

    Attributes
    ----------
    bundle_id:
        The case this authorisation applies to.
    actor:
        The reviewer performing the action. Must match the ``actor`` argument
        of the authority method being called, and must be a registered
        reviewer, since the signature is verified against their bound key.
    action:
        One of :data:`AUTHORIZED_ACTIONS`. The authority refuses an
        authorisation whose ``action`` is not the action being performed, so a
        ``dispute`` authorisation can never move a case to ``revoked``.
    nonce:
        A value that makes this authorisation single-use. The authority records
        spent authorisations and refuses a repeat.
    """

    bundle_id: str
    actor: str
    action: str
    nonce: str
    signed_at: str
    signature: str = ""
    key_id: str = ""

    def signing_material(self) -> bytes:
        """Return the exact bytes this signature covers.

        Every field that decides what the authorisation permits is inside it:
        the case, the actor, the action and the nonce. Changing any of them
        invalidates the signature.
        """
        return canonical_json(
            {
                "record": "certification_action_authorization",
                "bundle_id": self.bundle_id,
                "actor": self.actor,
                "action": self.action,
                "nonce": self.nonce,
                "signed_at": self.signed_at,
            }
        )

    @classmethod
    def create(
        cls,
        signer: Signer,
        *,
        bundle_id: str,
        actor: str,
        action: str,
        nonce: Optional[str] = None,
        signed_at: Optional[str] = None,
    ) -> "ActionAuthorization":
        """Build and sign an authorisation with the actor's own key.

        ``nonce`` defaults to a fresh 128-bit random value from
        :mod:`secrets`. Pass an explicit one for a deterministic test; two
        authorisations that share a nonce are two attempts to spend the same
        single-use token, and the authority accepts only the first.

        Raises
        ------
        CertificationError
            If ``action`` is not one of :data:`AUTHORIZED_ACTIONS`.
        """
        if action not in AUTHORIZED_ACTIONS:
            raise CertificationError(
                f"action must be one of {AUTHORIZED_ACTIONS}, got {action!r}"
            )
        unsigned = cls(
            bundle_id=bundle_id,
            actor=actor,
            action=action,
            nonce=nonce if nonce is not None else secrets.token_hex(16),
            signed_at=signed_at if signed_at is not None else utc_now_iso(),
        )
        return cls(
            bundle_id=unsigned.bundle_id,
            actor=unsigned.actor,
            action=unsigned.action,
            nonce=unsigned.nonce,
            signed_at=unsigned.signed_at,
            signature=signer.sign(unsigned.signing_material()),
            key_id=signer.key_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "bundle_id": self.bundle_id,
            "actor": self.actor,
            "action": self.action,
            "nonce": self.nonce,
            "signed_at": self.signed_at,
            "signature": self.signature,
            "key_id": self.key_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActionAuthorization":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            bundle_id=str(data["bundle_id"]),
            actor=str(data["actor"]),
            action=str(data["action"]),
            nonce=str(data.get("nonce", "")),
            signed_at=str(data.get("signed_at", "")),
            signature=str(data.get("signature", "")),
            key_id=str(data.get("key_id", "")),
        )


__all__ = ["AUTHORIZED_ACTIONS", "ActionAuthorization"]

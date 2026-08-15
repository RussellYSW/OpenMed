"""Institution-bound reviewer keys and the signatures they produce.

A certification signature is only meaningful if the verifier knows *which
institution* it came from, so keys are registered against a reviewer identity
that carries an institution. The keyring is the trust anchor here: whoever
controls registration controls which institutions exist.

SECURITY NOTE: registration is local and unauthenticated -- this module models
the policy layer, not a PKI. In a real deployment the institution binding would
come from an organisational certificate (or a federation membership service),
and revocation would be checked against that authority rather than an in-memory
set.

SECURITY NOTE: a keyring built by :meth:`ReviewerKeyring.register` without an
explicit signer holds every reviewer's *private* key, so :meth:`~ReviewerKeyring.sign`
can mint any registered reviewer's signature. That is a convenience for demos
and tests, not a model of reviewers signing remotely. Use
:meth:`ReviewerKeyring.verify_only` to build the verifying half, and have
reviewers hand back detached signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from trustfed.certification.actions import ActionAuthorization
from trustfed.certification.errors import KeyringReadOnlyError, UnknownReviewerError
from trustfed.ledger.crypto import Signer, Verifier, canonical_json, default_signer


@dataclass(frozen=True)
class Reviewer:
    """A person (or office) entitled to sign certification decisions.

    ``declared_conflicts`` holds bundle ids or institution names the reviewer
    has self-declared a conflict with; conflict-of-interest rules consult it.
    """

    reviewer_id: str
    institution: str
    role: str = "reviewer"
    declared_conflicts: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "reviewer_id": self.reviewer_id,
            "institution": self.institution,
            "role": self.role,
            "declared_conflicts": list(self.declared_conflicts),
        }


@dataclass(frozen=True)
class ReviewSignature:
    """One reviewer's signed decision on one bundle.

    ``decision`` is ``"approve"``, ``"reject"`` or ``"block"``. A ``block`` is
    the strong form used to open a dispute; a ``reject`` withholds this
    reviewer's approval without halting the case.
    """

    bundle_id: str
    reviewer_id: str
    institution: str
    decision: str
    statement: str
    signed_at: str
    signature: str
    key_id: str

    def signing_material(self) -> bytes:
        """Return the exact bytes this signature covers."""
        return canonical_json(
            {
                "bundle_id": self.bundle_id,
                "reviewer_id": self.reviewer_id,
                "institution": self.institution,
                "decision": self.decision,
                "statement": self.statement,
                "signed_at": self.signed_at,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "bundle_id": self.bundle_id,
            "reviewer_id": self.reviewer_id,
            "institution": self.institution,
            "decision": self.decision,
            "statement": self.statement,
            "signed_at": self.signed_at,
            "signature": self.signature,
            "key_id": self.key_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewSignature":
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            bundle_id=str(data["bundle_id"]),
            reviewer_id=str(data["reviewer_id"]),
            institution=str(data["institution"]),
            decision=str(data["decision"]),
            statement=str(data.get("statement", "")),
            signed_at=str(data.get("signed_at", "")),
            signature=str(data.get("signature", "")),
            key_id=str(data.get("key_id", "")),
        )


class ReviewerKeyring:
    """Registry of reviewers and the keys bound to them.

    Parameters
    ----------
    seed:
        Used to derive a deterministic key when :meth:`register` is called
        without an explicit signer. Demo and test convenience only -- a derived
        key is a published key.
    """

    def __init__(self, *, seed: bytes = b"trustfed-reviewer") -> None:
        self._seed = seed
        self._reviewers: Dict[str, Reviewer] = {}
        self._signers: Dict[str, Signer] = {}
        self._verifiers: Dict[str, Verifier] = {}
        self._revoked: set = set()
        self._sign_disabled = False

    @property
    def can_sign(self) -> bool:
        """Return ``True`` iff this keyring is able to produce signatures."""
        return not self._sign_disabled

    def verify_only(self) -> "ReviewerKeyring":
        """Return a copy of this keyring that holds no private keys.

        Same reviewers, same institution bindings, same revocations, same
        verifying keys -- but :meth:`sign` raises
        :class:`~trustfed.certification.errors.KeyringReadOnlyError`, and no
        signer object is carried over. Give this to a
        :class:`~trustfed.certification.authority.CertificationAuthority` when
        reviewers sign elsewhere and submit detached signatures via
        :meth:`~trustfed.certification.authority.CertificationAuthority.submit_signature`:
        the authority then structurally cannot mint a reviewer's approval,
        rather than merely not doing so.

        With the HMAC fallback signer the *verifying* key is the signing key,
        so a verify-only keyring built over HMAC keys can still forge in
        principle. Register Ed25519 verifiers (or a real PKI) for that
        separation to be real; see :mod:`trustfed.ledger.crypto`.
        """
        ring = ReviewerKeyring(seed=self._seed)
        ring._reviewers = dict(self._reviewers)
        ring._verifiers = dict(self._verifiers)
        ring._revoked = set(self._revoked)
        ring._signers = {}
        ring._sign_disabled = True
        return ring

    def register(
        self,
        reviewer: Reviewer,
        signer: Optional[Signer] = None,
        *,
        verifier: Optional[Verifier] = None,
    ) -> Reviewer:
        """Register ``reviewer`` and bind a key to them.

        Re-registering an existing reviewer id replaces the binding, which is
        how key rotation is modelled here. On a :meth:`verify_only` keyring an
        explicit ``verifier`` is required, since no signing key may be derived.
        """
        if self._sign_disabled:
            if verifier is None:
                raise KeyringReadOnlyError(
                    "a verify-only keyring can only register an explicit verifier"
                )
            self._reviewers[reviewer.reviewer_id] = reviewer
            self._verifiers[reviewer.reviewer_id] = verifier
            self._revoked.discard(reviewer.reviewer_id)
            return reviewer
        if signer is None:
            signer = default_signer(
                self._seed + b"|" + reviewer.reviewer_id.encode("utf-8")
            )
        self._reviewers[reviewer.reviewer_id] = reviewer
        self._signers[reviewer.reviewer_id] = signer
        self._verifiers[reviewer.reviewer_id] = (
            verifier if verifier is not None else signer.verifier()
        )
        self._revoked.discard(reviewer.reviewer_id)
        return reviewer

    def revoke(self, reviewer_id: str) -> None:
        """Mark a reviewer's key as no longer acceptable."""
        self._require(reviewer_id)
        self._revoked.add(reviewer_id)

    def is_revoked(self, reviewer_id: str) -> bool:
        """Return ``True`` iff this reviewer's key has been revoked."""
        return reviewer_id in self._revoked

    def reviewer(self, reviewer_id: str) -> Reviewer:
        """Return the registered reviewer record."""
        return self._require(reviewer_id)

    def institution_of(self, reviewer_id: str) -> str:
        """Return the institution bound to ``reviewer_id``."""
        return self._require(reviewer_id).institution

    def institutions(self) -> Tuple[str, ...]:
        """Return every institution with at least one registered reviewer."""
        return tuple(sorted({r.institution for r in self._reviewers.values()}))

    def sign(self, reviewer_id: str, signature: ReviewSignature) -> ReviewSignature:
        """Return ``signature`` with the reviewer's signature and key id filled in.

        SECURITY NOTE: this signs *as* the reviewer using a key this keyring
        holds. Any caller who knows a registered reviewer id can therefore
        produce that reviewer's signature. It exists so demos and tests can
        exercise the process in one process; a reviewer signing remotely would
        hold their own key and hand back a detached
        :class:`ReviewSignature`. See :meth:`verify_only`.

        Raises
        ------
        KeyringReadOnlyError
            If this keyring holds no private keys.
        """
        self._require(reviewer_id)
        if self._sign_disabled or reviewer_id not in self._signers:
            raise KeyringReadOnlyError(
                f"this keyring holds no signing key for {reviewer_id!r}; "
                "signatures must be produced by the reviewer and submitted "
                "via CertificationAuthority.submit_signature"
            )
        signer = self._signers[reviewer_id]
        return ReviewSignature(
            bundle_id=signature.bundle_id,
            reviewer_id=signature.reviewer_id,
            institution=signature.institution,
            decision=signature.decision,
            statement=signature.statement,
            signed_at=signature.signed_at,
            signature=signer.sign(signature.signing_material()),
            key_id=signer.key_id,
        )

    def sign_action(
        self, reviewer_id: str, authorization: ActionAuthorization
    ) -> ActionAuthorization:
        """Return ``authorization`` signed with the reviewer's registered key.

        SECURITY NOTE: the same caveat as :meth:`sign` applies, and it bites
        harder here, because an action authorisation moves the case. This
        signs *as* the reviewer with a key this keyring holds, so any caller
        who knows a registered reviewer id can mint one. It exists so demos and
        tests can drive the dispute path in one process. A real actor holds
        their own key and builds the record with
        :meth:`~trustfed.certification.actions.ActionAuthorization.create`.

        Raises
        ------
        KeyringReadOnlyError
            If this keyring holds no signing key for the reviewer.
        """
        self._require(reviewer_id)
        if self._sign_disabled or reviewer_id not in self._signers:
            raise KeyringReadOnlyError(
                f"this keyring holds no signing key for {reviewer_id!r}; "
                "action authorisations must be produced by the actor"
            )
        signer = self._signers[reviewer_id]
        return ActionAuthorization(
            bundle_id=authorization.bundle_id,
            actor=authorization.actor,
            action=authorization.action,
            nonce=authorization.nonce,
            signed_at=authorization.signed_at,
            signature=signer.sign(authorization.signing_material()),
            key_id=signer.key_id,
        )

    def verify_action(self, authorization: ActionAuthorization) -> bool:
        """Return ``True`` iff the authorisation verifies and the key is current.

        Checks that the named actor is registered and not revoked, that the
        recorded key id (when present) is the one bound to them now, and that
        the signature verifies over
        :meth:`~trustfed.certification.actions.ActionAuthorization.signing_material`
        -- which covers the action, so this cannot accept a record signed for a
        different action. It says nothing about whether the authorisation was
        already spent; that is the authority's job.
        """
        if authorization.actor not in self._reviewers:
            return False
        if authorization.actor in self._revoked:
            return False
        if not authorization.signature:
            return False
        verifier = self._verifiers[authorization.actor]
        if authorization.key_id and authorization.key_id != verifier.key_id:
            return False
        return verifier.verify(
            authorization.signing_material(), authorization.signature
        )

    def verify(self, signature: ReviewSignature) -> bool:
        """Return ``True`` iff the signature verifies and the key is current.

        Checks three things: the reviewer is registered and not revoked, the
        institution recorded in the signature matches the registered binding,
        and the signature itself verifies under the bound key.
        """
        reviewer = self._reviewers.get(signature.reviewer_id)
        if reviewer is None or signature.reviewer_id in self._revoked:
            return False
        if reviewer.institution != signature.institution:
            return False
        verifier = self._verifiers[signature.reviewer_id]
        if signature.key_id and signature.key_id != verifier.key_id:
            return False
        return verifier.verify(signature.signing_material(), signature.signature)

    def _require(self, reviewer_id: str) -> Reviewer:
        try:
            return self._reviewers[reviewer_id]
        except KeyError as exc:
            raise UnknownReviewerError(
                f"no key registered for reviewer {reviewer_id!r}"
            ) from exc

    def __contains__(self, reviewer_id: object) -> bool:
        """Return ``True`` iff a reviewer with that id is registered."""
        return reviewer_id in self._reviewers

    def __len__(self) -> int:
        """Return the number of registered reviewers."""
        return len(self._reviewers)


__all__ = ["Reviewer", "ReviewSignature", "ReviewerKeyring"]

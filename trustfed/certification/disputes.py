"""Dispute, appeal, remediation and revocation actions on a certification case.

Mixed into :class:`~trustfed.certification.authority.CertificationAuthority`.
Every action here appends to the decision log before it changes state, and every
state change is checked against the machine in
:mod:`trustfed.certification.states`, so an illegal move is a typed error rather
than an inconsistent record.

TRUST MODEL: ``actor``, ``raised_by`` and friends are *asserted* identities, not
authenticated ones. Unless a signed
:class:`~trustfed.certification.actions.ActionAuthorization` is supplied, any
caller can pass any string and the log will faithfully record whatever they
passed. This matters most for :meth:`DisputeActionsMixin.revoke` and a refused
appeal, because ``revoked`` is terminal -- a revoked bundle can never be
certified again. Every payload written here therefore carries
``actor_authenticated``, so a reader of the decision log can tell an
authenticated decision from a claimed one, and the authority can be built with
``require_signed_actions=True`` to refuse the unauthenticated form outright.

An authorisation is *not* a review signature. A
:class:`~trustfed.certification.keys.ReviewSignature` is published into the
decision log the moment it is filed and says nothing about what should be done
to the case, so accepting one here would let anyone who can read the log replay
an honest reviewer's blocking review as authority for a terminal revocation.
This module refuses that record outright; see
:mod:`trustfed.certification.actions`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Set, Tuple

from trustfed.certification.actions import ActionAuthorization
from trustfed.certification.case import CertificationCase
from trustfed.certification.errors import (
    CertificationError,
    SignatureRejectedError,
    UnknownReviewerError,
)
from trustfed.certification.keys import ReviewSignature
from trustfed.certification.states import CertificationState

#: Decision a review signature must carry to open a dispute through
#: :meth:`~trustfed.certification.authority.CertificationAuthority.submit_signature`.
BLOCKING_DECISION = "block"


class DisputeActionsMixin:
    """Dispute/appeal half of the certification authority.

    Assumes the host class provides ``case``, ``_log``, ``_transition``,
    ``_keyring``, ``_ledger``, ``_cases``, ``_spent_authorizations`` and
    ``_require_signed_actions``.

    Each action takes an optional ``authorization``: an
    :class:`~trustfed.certification.actions.ActionAuthorization` over the same
    bundle, naming the same actor and *the same action*, produced by that
    actor's registered key. When it is present the authority verifies it,
    spends it (it is single-use) and records ``actor_authenticated: True``.
    When it is absent the action still proceeds -- unless the authority was
    built with ``require_signed_actions=True`` and the action is one of the
    terminal ones -- but is recorded as unauthenticated.
    """

    def _authenticate_actor(
        self,
        bundle_id: str,
        actor: str,
        authorization: Optional[ActionAuthorization],
        *,
        action: str,
        required: bool = False,
    ) -> Tuple[Dict[str, Any], bool]:
        """Return ledger ``extra`` fields describing how ``actor`` was checked.

        Raises
        ------
        SignatureRejectedError
            If a :class:`~trustfed.certification.keys.ReviewSignature` (or any
            other object) was passed instead of an
            :class:`~trustfed.certification.actions.ActionAuthorization`; if
            the authorisation does not cover this bundle, this actor or *this
            action*; if it does not verify under the actor's registered key; if
            it has already been spent; or if none was supplied and this
            authority requires one.
        """
        if authorization is None:
            if required and getattr(self, "_require_signed_actions", False):
                raise SignatureRejectedError(
                    f"{action} on {bundle_id} requires a signed "
                    "ActionAuthorization; this authority was built with "
                    "require_signed_actions=True"
                )
            extra: Dict[str, Any] = {
                "actor_authenticated": False,
                "actor_institution": self._institution_of(actor),
            }
            return extra, False

        if isinstance(authorization, ReviewSignature):
            raise SignatureRejectedError(
                "a ReviewSignature does not authorise a case action: it is "
                "signed over a review decision, not over an action, and it is "
                "published into the decision log as soon as it is filed, so "
                "accepting it here would let any reader of the log replay a "
                f"reviewer's blocking review as authority for {action!r}. "
                "Supply an ActionAuthorization instead."
            )
        if not isinstance(authorization, ActionAuthorization):
            raise SignatureRejectedError(
                "expected an ActionAuthorization, got "
                f"{type(authorization).__name__}"
            )
        if authorization.bundle_id != bundle_id:
            raise SignatureRejectedError(
                f"authorization covers {authorization.bundle_id}, not {bundle_id}"
            )
        if authorization.actor != actor:
            raise SignatureRejectedError(
                f"authorization is from {authorization.actor!r}, not the named "
                f"actor {actor!r}"
            )
        if authorization.action != action:
            raise SignatureRejectedError(
                f"authorization permits {authorization.action!r}, not "
                f"{action!r}; an authorisation is bound to one action"
            )
        if not self._keyring.verify_action(authorization):
            raise SignatureRejectedError(
                f"authorization from {actor} did not verify under their bound key"
            )
        self._spend_authorization(bundle_id, authorization)
        return (
            {
                "actor_authenticated": True,
                "actor_institution": self._institution_of(actor),
                "authorization": authorization.to_dict(),
            },
            True,
        )

    def _spend_authorization(
        self, bundle_id: str, authorization: ActionAuthorization
    ) -> None:
        """Consume a single-use authorisation, refusing one already spent.

        Three places are checked, because a signature that verifies is a bearer
        token once it has been seen:

        1. the in-process set of authorisations this authority has spent;
        2. the decision log, so a replay still fails after a restart, or
           against an authority rebuilt over a persisted ledger;
        3. the recorded review signatures on the case, so a signature lifted
           from a filed review cannot be re-presented in this envelope. (The
           type check in :meth:`_authenticate_actor` and the distinct signing
           material already make that impossible; this is belt and braces.)

        Raises
        ------
        SignatureRejectedError
            If this authorisation has been used before.
        """
        spent: Set[Tuple[str, str, str, str]] = self._spent_authorizations
        key = (
            bundle_id,
            authorization.actor,
            authorization.action,
            authorization.nonce,
        )
        if key in spent:
            raise SignatureRejectedError(
                f"this authorization has already been spent on {bundle_id}; "
                "an authorisation is single-use, so a repeated action needs a "
                "newly signed one"
            )
        for block in self._ledger:
            prior = block.payload.get("authorization")
            if (
                isinstance(prior, Mapping)
                and prior.get("signature") == authorization.signature
            ):
                raise SignatureRejectedError(
                    f"this authorization was already recorded in the decision "
                    f"log (block {block.index}); it cannot be replayed"
                )
        case = self._cases.get(bundle_id)
        if case is not None:
            for filed in case.signatures:
                if filed.signature and filed.signature == authorization.signature:
                    raise SignatureRejectedError(
                        "this signature was filed as a review on "
                        f"{bundle_id}; a review signature is not an action "
                        "authorisation"
                    )
        spent.add(key)

    def _institution_of(self, actor: str) -> Optional[str]:
        """Return the actor's registered institution, or ``None`` if unknown.

        An unregistered actor is not an error on this path -- a site lead or an
        ethics office may legitimately raise a dispute without being a
        registered reviewer -- but the log records that nothing bound the name
        to an institution.
        """
        try:
            return self._keyring.reviewer(actor).institution
        except UnknownReviewerError:
            return None

    def dispute(
        self,
        bundle_id: str,
        *,
        raised_by: str,
        reason: str,
        authorization: Optional[ActionAuthorization] = None,
    ) -> CertificationCase:
        """Block a case because a dispute was raised against it.

        ``raised_by`` is an asserted identity unless ``authorization`` -- an
        :class:`~trustfed.certification.actions.ActionAuthorization` for the
        ``"dispute"`` action, signed by ``raised_by`` -- is given; see the class
        docstring. Blocking is recoverable (via :meth:`remediate`), so this is
        the least consequential of the actions here.
        """
        case = self.case(bundle_id)
        extra, _ = self._authenticate_actor(
            bundle_id, raised_by, authorization, action="dispute"
        )
        self._transition(
            case,
            CertificationState.BLOCKED,
            "dispute_raised",
            raised_by,
            detail=reason,
            extra=extra,
        )
        return case

    def remediate(
        self,
        bundle_id: str,
        *,
        actor: str,
        note: str,
        clear_signatures: bool = True,
        authorization: Optional[ActionAuthorization] = None,
    ) -> CertificationCase:
        """Record a remediation of a blocked case.

        Prior approvals are cleared by default: they were given for content or
        conduct that has since changed, so re-review is required before the
        threshold can be met again. ``actor`` is asserted, not authenticated,
        unless an ``authorization`` for the ``"remediate"`` action is given;
        the move is recoverable either way, since a remediated case must be
        re-reviewed to be certified.
        """
        case = self.case(bundle_id)
        cleared = len(case.signatures) if clear_signatures else 0
        extra, _ = self._authenticate_actor(
            bundle_id, actor, authorization, action="remediate"
        )
        extra["cleared_signatures"] = cleared
        self._transition(
            case,
            CertificationState.REMEDIATED,
            "remediated",
            actor,
            detail=note,
            extra=extra,
        )
        if clear_signatures:
            case.signatures = []
        return case

    def appeal(
        self,
        bundle_id: str,
        *,
        actor: str,
        grounds: str,
        authorization: Optional[ActionAuthorization] = None,
    ) -> CertificationCase:
        """Record an appeal against a blocked case. Does not change state.

        ``actor`` is asserted, not authenticated, unless an ``authorization``
        for the ``"appeal"`` action is given. Since nothing moves, an
        unauthenticated appeal costs the case nothing beyond a log entry.
        """
        case = self.case(bundle_id)
        if case.state is not CertificationState.BLOCKED:
            raise CertificationError(
                f"appeals apply to blocked cases; {bundle_id} is {case.state.value}"
            )
        extra, _ = self._authenticate_actor(
            bundle_id, actor, authorization, action="appeal"
        )
        self._log(
            bundle_id,
            "appeal_filed",
            actor,
            from_state=case.state,
            to_state=case.state,
            detail=grounds,
            extra=extra,
        )
        return case

    def resolve_appeal(
        self,
        bundle_id: str,
        *,
        actor: str,
        upheld: bool,
        note: str = "",
        authorization: Optional[ActionAuthorization] = None,
    ) -> CertificationCase:
        """Resolve an appeal: uphold it (remediated) or refuse it (revoked).

        Refusing an appeal is terminal, so it is treated like :meth:`revoke`:
        an authority built with ``require_signed_actions=True`` refuses to do
        it without a verified ``"appeal_refused"`` authorisation from ``actor``.

        Note that the two outcomes need *different* authorisations, because an
        authorisation is bound to one action: upholding delegates to
        :meth:`remediate` and therefore wants a ``"remediate"`` authorisation,
        while refusing wants an ``"appeal_refused"`` one. Signing for the
        outcome you intend is the point -- an authorisation to remediate can
        never be spent to revoke.
        """
        case = self.case(bundle_id)
        if upheld:
            return self.remediate(
                bundle_id,
                actor=actor,
                note=note or "appeal upheld",
                authorization=authorization,
            )
        extra, _ = self._authenticate_actor(
            bundle_id, actor, authorization, action="appeal_refused", required=True
        )
        self._transition(
            case,
            CertificationState.REVOKED,
            "appeal_refused",
            actor,
            detail=note or "appeal refused",
            extra=extra,
        )
        return case

    def revoke(
        self,
        bundle_id: str,
        *,
        actor: str,
        reason: str,
        authorization: Optional[ActionAuthorization] = None,
    ) -> CertificationCase:
        """Revoke a case. Terminal: a revoked bundle can never be certified.

        This is the most consequential action in the state machine, so it is
        the one worth signing. ``authorization`` must be an
        :class:`~trustfed.certification.actions.ActionAuthorization` for the
        ``"revoke"`` action, signed by ``actor``, and it is spent here: the
        same one cannot revoke twice, and no other signed record -- a blocking
        review included -- is accepted in its place. Without it the ``actor``
        is a claim and the decision log records ``actor_authenticated: False``;
        build the authority with ``require_signed_actions=True`` to refuse that
        form.
        """
        case = self.case(bundle_id)
        extra, _ = self._authenticate_actor(
            bundle_id, actor, authorization, action="revoke", required=True
        )
        self._transition(
            case,
            CertificationState.REVOKED,
            "revoked",
            actor,
            detail=reason,
            extra=extra,
        )
        return case


__all__ = ["BLOCKING_DECISION", "DisputeActionsMixin"]

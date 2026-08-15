"""Multi-party certification: cases, signed reviews, and the decision log.

The authority holds one *case* per bundle. Every action -- submission, a signed
review, a refusal on conflict-of-interest grounds, certification, a dispute, a
remediation, an appeal, a revocation -- is appended to the ledger before the
case state changes, so the record of how a decision was reached cannot be
edited afterwards without breaking the chain.

The authority decides *governance*, not clinical safety: it records who signed
what, from which institution, and whether the policy threshold was met.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Set, Tuple

from trustfed.certification.case import CertificationCase
from trustfed.certification.decisionlog import EVENT_CERTIFICATION, DecisionLogMixin
from trustfed.certification.disputes import DisputeActionsMixin
from trustfed.certification.errors import (
    CertificationError,
    ConflictOfInterestError,
    SignatureRejectedError,
    ThresholdNotMetError,
    UnknownCaseError,
)
from trustfed.certification.keys import ReviewSignature, ReviewerKeyring
from trustfed.certification.policy import ThresholdOutcome, ThresholdPolicy
from trustfed.certification.submission import SubmissionMixin
from trustfed.certification.states import CertificationState, assert_transition
from trustfed.ledger.backend import LedgerBackend
from trustfed.ledger.block import utc_now_iso
from trustfed.ledger.memory import InMemoryLedger

#: Decisions a reviewer may sign.
VALID_DECISIONS: Tuple[str, ...] = ("approve", "reject", "block")


class CertificationAuthority(SubmissionMixin, DecisionLogMixin, DisputeActionsMixin):
    """Runs the certification, dispute and appeal process for model bundles.

    TRUST MODEL -- read before quoting the threshold as an enforced control:

    * :meth:`submit_signature` is the real path. The authority verifies a
      detached signature produced elsewhere and never holds a signing key.
    * :meth:`review` is a demo/test convenience that signs *on the reviewer's
      behalf* with a key the authority holds, so any caller who knows a
      reviewer id can mint that reviewer's approval.
    * :meth:`submit` derives the owner institution from the bundle when given
      one, and marks the case ``self_asserted`` when given only a bundle id
      plus a free-string owner.
    * The dispute-side actions in :class:`DisputeActionsMixin` take an actor
      string that is asserted, not authenticated, unless a signed
      :class:`~trustfed.certification.actions.ActionAuthorization` for that
      exact action is supplied; see that class.

    Parameters
    ----------
    keyring:
        Institution-bound reviewer keys. Pass
        :meth:`~trustfed.certification.keys.ReviewerKeyring.verify_only` output
        to build an authority that structurally cannot sign for a reviewer.
    ledger:
        Append-only decision log. Defaults to a non-durable
        :class:`~trustfed.ledger.memory.InMemoryLedger`.
    policy:
        Threshold and conflict-of-interest policy.
    clock:
        Callable returning an ISO-8601 timestamp; injectable for tests.
    require_manual:
        Refuse submissions without a valid fine-tuning manual. Default False,
        because the manual may legitimately arrive with the first revision.
    require_signed_actions:
        Refuse an unauthorised :meth:`~DisputeActionsMixin.revoke` or refused
        appeal -- that is, one without a verified, single-use
        :class:`~trustfed.certification.actions.ActionAuthorization`. Default
        False so the demo and tests can drive the state machine with a plain
        actor string; set it True in any deployment where revocation matters,
        since revocation is terminal.
    """

    def __init__(
        self,
        keyring: ReviewerKeyring,
        ledger: Optional[LedgerBackend] = None,
        *,
        policy: Optional[ThresholdPolicy] = None,
        clock: Optional[Callable[[], str]] = None,
        require_manual: bool = False,
        require_signed_actions: bool = False,
    ) -> None:
        self._keyring = keyring
        self._ledger: LedgerBackend = ledger if ledger is not None else InMemoryLedger()
        self._policy = policy if policy is not None else ThresholdPolicy()
        self._clock: Callable[[], str] = clock if clock is not None else utc_now_iso
        self._require_manual = require_manual
        self._require_signed_actions = require_signed_actions
        self._cases: Dict[str, CertificationCase] = {}
        # Action authorisations already spent, as
        # (bundle_id, actor, action, nonce). The decision log is also scanned
        # on every spend, so this set is a fast path rather than the whole
        # defence -- see DisputeActionsMixin._spend_authorization.
        self._spent_authorizations: Set[Tuple[str, str, str, str]] = set()

    # ------------------------------------------------------------------ properties

    @property
    def keyring(self) -> ReviewerKeyring:
        """The reviewer keyring this authority verifies signatures against."""
        return self._keyring

    @property
    def policy(self) -> ThresholdPolicy:
        """The threshold / conflict policy in force."""
        return self._policy

    @property
    def ledger(self) -> LedgerBackend:
        """The append-only decision log."""
        return self._ledger

    def case(self, bundle_id: str) -> CertificationCase:
        """Return the case for ``bundle_id``.

        Raises
        ------
        UnknownCaseError
            If no case has been submitted for that bundle.
        """
        try:
            return self._cases[bundle_id]
        except KeyError as exc:
            raise UnknownCaseError(f"no certification case for {bundle_id}") from exc

    def state(self, bundle_id: str) -> CertificationState:
        """Return the current state of ``bundle_id``'s case."""
        return self.case(bundle_id).state

    def is_certified_base(self, bundle_id: str) -> bool:
        """Return ``True`` iff the bundle is currently a certified base model."""
        return (
            bundle_id in self._cases
            and self._cases[bundle_id].state is CertificationState.CERTIFIED
        )

    # ------------------------------------------------------------------- actions

    def submit_signature(self, signature: ReviewSignature) -> ReviewSignature:
        """Accept a detached, already-signed review from an external reviewer.

        This is the path a real deployment should use. The authority never
        touches a private key here: it looks the reviewer up, applies the
        conflict-of-interest rules, verifies the detached signature against the
        reviewer's registered *verifying* key, and only then records it. The
        signature covers the bundle id, reviewer id, institution, decision,
        statement and timestamp (see
        :meth:`~trustfed.certification.keys.ReviewSignature.signing_material`),
        so none of those can be altered in transit, and
        :meth:`~trustfed.certification.keys.ReviewerKeyring.verify` refuses a
        signature that claims an institution its key is not bound to.

        Pair it with
        :meth:`~trustfed.certification.keys.ReviewerKeyring.verify_only`, which
        gives the authority a keyring holding no private keys at all, so the
        code path that could mint a reviewer's approval does not exist in the
        verifying process.

        A ``"block"`` decision is accepted against a *certified* case as well,
        and moves it straight to ``blocked``. Post-certification safety signals
        are the ones most worth binding to a reviewer's key, and refusing them
        here would push the objection onto
        :meth:`~DisputeActionsMixin.dispute`, whose actor is asserted unless a
        separate authorisation is attached.

        What it still does not do: authenticate the *transport*, or establish
        the institution binding itself. The keyring is the trust anchor and its
        registration is local and unauthenticated -- see
        :mod:`trustfed.certification.keys`.

        Raises
        ------
        UnknownReviewerError
            If the signing reviewer is not registered.
        ConflictOfInterestError
            If a conflict rule objects to this reviewer (the refusal is logged).
        SignatureRejectedError
            If the signature does not verify under the reviewer's bound key.
        InvalidTransitionError
            If the case is in a state that cannot accept reviews.
        """
        if signature.decision not in VALID_DECISIONS:
            raise CertificationError(
                f"decision must be one of {VALID_DECISIONS}, got "
                f"{signature.decision!r}"
            )
        bundle_id = signature.bundle_id
        reviewer_id = signature.reviewer_id
        case = self.case(bundle_id)
        reviewer = self._keyring.reviewer(reviewer_id)

        conflict = self._policy.check_conflicts(reviewer, case.facts)
        if conflict is not None:
            code, detail = conflict
            self._log(
                bundle_id,
                "review_refused",
                reviewer_id,
                from_state=case.state,
                to_state=case.state,
                detail=detail,
                extra={"reason_code": code, "institution": reviewer.institution},
            )
            raise ConflictOfInterestError(code, detail)

        if not self._keyring.verify(signature):
            raise SignatureRejectedError(
                f"signature from {reviewer_id} did not verify under their bound key"
            )

        blocking = signature.decision == "block"
        if case.state is CertificationState.CERTIFIED and blocking:
            # A safety signal about an already-certified model is the highest
            # stakes review there is, and the state machine allows
            # certified -> blocked. Reopening review first would not: it would
            # demand certified -> under_review, which is not a legal move, and
            # would force the objection onto the unauthenticated dispute()
            # path. The block below moves the case straight to BLOCKED.
            pass
        elif case.state in (
            CertificationState.SUBMITTED,
            CertificationState.REMEDIATED,
        ):
            self._transition(
                case,
                CertificationState.UNDER_REVIEW,
                "review_opened",
                reviewer_id,
                detail="first review received",
            )
        elif case.state is not CertificationState.UNDER_REVIEW:
            assert_transition(case.state, CertificationState.UNDER_REVIEW)

        case.signatures.append(signature)
        self._log(
            bundle_id,
            "review_signed",
            reviewer_id,
            from_state=case.state,
            to_state=case.state,
            detail=signature.statement,
            extra={"signature": signature.to_dict()},
        )
        if blocking:
            self._transition(
                case,
                CertificationState.BLOCKED,
                "blocked_by_review",
                reviewer_id,
                detail=signature.statement or "blocking objection filed",
            )
        return signature

    def review(
        self,
        bundle_id: str,
        reviewer_id: str,
        decision: str,
        *,
        statement: str = "",
    ) -> ReviewSignature:
        """Sign and record a reviewer's decision -- demo and test convenience.

        SECURITY NOTE: this method signs on the reviewer's behalf using a key
        the authority holds, so it does not model a reviewer signing remotely;
        any caller of ``review()`` can produce any registered reviewer's
        approval. The subsequent verification step is the same process checking
        its own signature, which proves nothing about who decided. The k-of-n
        threshold is therefore an *organisational* control expressed in code,
        not a cryptographic one, whenever this path is used.

        Use :meth:`submit_signature` for anything where the reviewer and the
        authority are different principals.

        Refuses -- and logs the refusal -- when the reviewer is conflicted, for
        example when their institution owns the model.

        Raises
        ------
        ConflictOfInterestError
            If a conflict rule objects to this reviewer.
        SignatureRejectedError
            If the produced signature does not verify (revoked or rotated key).
        InvalidTransitionError
            If the case is in a state that cannot accept reviews.
        """
        if decision not in VALID_DECISIONS:
            raise CertificationError(
                f"decision must be one of {VALID_DECISIONS}, got {decision!r}"
            )
        self.case(bundle_id)
        reviewer = self._keyring.reviewer(reviewer_id)
        signature = self._keyring.sign(
            reviewer_id,
            ReviewSignature(
                bundle_id=bundle_id,
                reviewer_id=reviewer_id,
                institution=reviewer.institution,
                decision=decision,
                statement=statement,
                signed_at=self._clock(),
                signature="",
                key_id="",
            ),
        )
        return self.submit_signature(signature)

    def evaluate(self, bundle_id: str) -> ThresholdOutcome:
        """Return the current threshold outcome without changing state.

        Only signatures that still verify under the keyring are counted, so a
        revoked reviewer's approval stops counting.
        """
        case = self.case(bundle_id)
        valid = [s for s in case.signatures if self._keyring.verify(s)]
        return self._policy.evaluate(valid)

    def certify(self, bundle_id: str, *, actor: str = "authority") -> CertificationCase:
        """Certify the bundle if the multi-institution threshold is met.

        Raises
        ------
        ThresholdNotMetError
            If the policy is not satisfied; the reason code is included.
        InvalidTransitionError
            If the case is not in a certifiable state.
        """
        case = self.case(bundle_id)
        # Refuse an illegal state move before spending effort on the threshold,
        # so a revoked case reports the state error rather than a vote count.
        assert_transition(case.state, CertificationState.CERTIFIED)
        outcome = self.evaluate(bundle_id)
        if not outcome.met:
            self._log(
                bundle_id,
                "certification_refused",
                actor,
                from_state=case.state,
                to_state=case.state,
                detail=outcome.detail,
                extra={"outcome": outcome.to_dict()},
            )
            raise ThresholdNotMetError(f"[{outcome.reason_code}] {outcome.detail}")
        self._transition(
            case,
            CertificationState.CERTIFIED,
            "certified",
            actor,
            detail=outcome.detail,
            extra={"outcome": outcome.to_dict(), "policy": self._policy.to_dict()},
        )
        return case

__all__ = [
    "EVENT_CERTIFICATION",
    "VALID_DECISIONS",
    "CertificationAuthority",
    "CertificationCase",
]

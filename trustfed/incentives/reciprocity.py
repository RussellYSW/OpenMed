"""Reciprocity: who is owed an evaluation, and why a request is refused.

An open commons collapses if every site wants its model evaluated and none wants
to run evaluations. The rule implemented here is deliberately simple and
inspectable: a site must have served as an evaluator at least N times before it
may request an evaluation. Every decision carries a machine-readable reason code
so the refusal can be shown to the requesting site and audited later.

This is an incentive rule, not an access control: it decides queue fairness, not
who may read data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional

from trustfed.incentives.credit import CreditLedger
from trustfed.incentives.errors import ReciprocityRefused

#: Decision reason codes.
REASON_GRANTED = "granted"
REASON_EXEMPT = "exempt"
REASON_INSUFFICIENT_RECIPROCITY = "insufficient_reciprocity"
REASON_SELF_EVALUATION = "self_evaluation_refused"
REASON_UNKNOWN_REQUESTER = "unknown_requester"
REASON_INSUFFICIENT_STANDING = "insufficient_standing"


@dataclass(frozen=True)
class EvaluationRequest:
    """A request for another site to evaluate one's model."""

    requester: str
    evaluator: str
    bundle_id: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "requester": self.requester,
            "evaluator": self.evaluator,
            "bundle_id": self.bundle_id,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReciprocityDecision:
    """Accept or refuse an evaluation request, with a stated reason."""

    allowed: bool
    reason_code: str
    detail: str
    request: EvaluationRequest
    observed: int = 0
    required: int = 0

    def raise_if_refused(self) -> None:
        """Raise :class:`ReciprocityRefused` when the request was not granted."""
        if not self.allowed:
            raise ReciprocityRefused(self.reason_code, self.detail)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the decision."""
        return {
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "observed": self.observed,
            "required": self.required,
            "request": self.request.to_dict(),
        }


@dataclass(frozen=True)
class ReciprocityPolicy:
    """Decide whether a site may receive an evaluation.

    Parameters
    ----------
    min_evaluations_served:
        How many times the requester must have served as an evaluator.
    exempt_actors:
        Actors the rule does not apply to (e.g. a newly joined site inside its
        grace period, agreed in governance).
    min_voting_weight:
        Optional standing floor, applied in addition to the evaluation count.
    allow_self_evaluation:
        Whether a site may be its own evaluator. False by default: a site
        evaluating its own model is not an independent evaluation.
    require_attested_service:
        Count only ``evaluation_served`` events carrying a counterparty
        attestation that verified when it was recorded. Default False, which
        means the served count is **self-asserted**: a site may append its own
        ``evaluation_served`` credit through
        :meth:`~trustfed.incentives.credit.CreditLedger.record` and satisfy
        this rule without anyone confirming it happened. Set this True, and
        record service through
        :meth:`~trustfed.incentives.credit.CreditLedger.record_evaluation_served`
        with a signed attestation, for the rule to rest on a second party's
        word instead.
    """

    min_evaluations_served: int = 2
    exempt_actors: FrozenSet[str] = frozenset()
    min_voting_weight: Optional[int] = None
    allow_self_evaluation: bool = False
    require_attested_service: bool = False

    def __post_init__(self) -> None:
        if self.min_evaluations_served < 0:
            raise ValueError("min_evaluations_served must be >= 0")
        object.__setattr__(self, "exempt_actors", frozenset(self.exempt_actors))

    def decide(
        self, request: EvaluationRequest, credit: CreditLedger
    ) -> ReciprocityDecision:
        """Return an accept/refuse decision for ``request``.

        Checks, in order: self-evaluation, exemption, evaluations served, and
        (optionally) standing. The first failing check determines the reason
        code; a granted decision reports how many evaluations were observed.

        SCOPE NOTE: unless ``require_attested_service`` is set, the served
        count comes from entries the requesting site could have appended about
        itself -- the credit ledger authenticates nobody. A deployment that
        wants this rule to be more than an honour system needs an external
        attestation source; see
        :mod:`trustfed.incentives.attest`.
        """
        served = credit.served_evaluations(
            request.requester, attested_only=self.require_attested_service
        )

        if not self.allow_self_evaluation and request.requester == request.evaluator:
            return ReciprocityDecision(
                allowed=False,
                reason_code=REASON_SELF_EVALUATION,
                detail=(
                    f"{request.requester} cannot act as its own evaluator; "
                    "an independent site is required"
                ),
                request=request,
                observed=served,
                required=self.min_evaluations_served,
            )

        if request.requester in self.exempt_actors:
            return ReciprocityDecision(
                allowed=True,
                reason_code=REASON_EXEMPT,
                detail=f"{request.requester} is exempt from the reciprocity rule",
                request=request,
                observed=served,
                required=self.min_evaluations_served,
            )

        if served < self.min_evaluations_served:
            return ReciprocityDecision(
                allowed=False,
                reason_code=REASON_INSUFFICIENT_RECIPROCITY,
                detail=(
                    f"{request.requester} has served as evaluator {served} time(s); "
                    f"policy requires {self.min_evaluations_served} before an "
                    "evaluation may be requested"
                ),
                request=request,
                observed=served,
                required=self.min_evaluations_served,
            )

        if self.min_voting_weight is not None:
            try:
                weight = credit.standing(request.requester).voting_weight
            except KeyError:
                weight = 0
            if weight < self.min_voting_weight:
                return ReciprocityDecision(
                    allowed=False,
                    reason_code=REASON_INSUFFICIENT_STANDING,
                    detail=(
                        f"{request.requester} has voting weight {weight}; policy "
                        f"requires {self.min_voting_weight}"
                    ),
                    request=request,
                    observed=served,
                    required=self.min_evaluations_served,
                )

        return ReciprocityDecision(
            allowed=True,
            reason_code=REASON_GRANTED,
            detail=(
                f"{request.requester} has served as evaluator {served} time(s), "
                f"meeting the required {self.min_evaluations_served}"
            ),
            request=request,
            observed=served,
            required=self.min_evaluations_served,
        )

    def decide_and_record(
        self, request: EvaluationRequest, credit: CreditLedger
    ) -> ReciprocityDecision:
        """Decide, append the decision to the credit ledger, and return it."""
        decision = self.decide(request, credit)
        credit.record_decision(decision)
        return decision

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description of the policy."""
        return {
            "min_evaluations_served": self.min_evaluations_served,
            "exempt_actors": sorted(self.exempt_actors),
            "min_voting_weight": self.min_voting_weight,
            "allow_self_evaluation": self.allow_self_evaluation,
            "require_attested_service": self.require_attested_service,
        }


__all__ = [
    "REASON_EXEMPT",
    "REASON_GRANTED",
    "REASON_INSUFFICIENT_RECIPROCITY",
    "REASON_INSUFFICIENT_STANDING",
    "REASON_SELF_EVALUATION",
    "REASON_UNKNOWN_REQUESTER",
    "EvaluationRequest",
    "ReciprocityDecision",
    "ReciprocityPolicy",
]

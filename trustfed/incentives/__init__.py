"""Component 6 -- credit and reciprocity for a governed model commons.

Contribution is recorded as credit events on the tamper-evident ledger
(:class:`CreditLedger`); releases get citable, DOI-style identifiers and named
attribution (:class:`ReleaseAttribution`); accumulated credit maps to a standing
tier and a governance voting weight (:class:`StandingReport`); and
:class:`ReciprocityPolicy` accepts or refuses an evaluation request with a
machine-readable reason -- by default, a site must have served as an evaluator N
times before it may ask for one.

Three honesty notes. The identifiers minted here are DOI-*style* but locally
minted and **not registered DOIs**. The credit weights, tiers and voting weights
are governance defaults shipped for illustration, not empirically derived
values; a deployment is expected to replace them. And the credit ledger
authenticates nobody: appending is open to any caller, so a site can record
credit for itself and the hash chain will not object -- what the chain detects
is a *retroactive edit* to something already recorded. Use
:meth:`CreditLedger.record_evaluation_served` with a
:class:`ServiceAttestation` for the one kind where a second party can vouch,
and :class:`ReciprocityPolicy` with ``require_attested_service=True`` to make
the rule depend on it.
"""

from __future__ import annotations

from trustfed.incentives.attest import CounterpartyRegistry, ServiceAttestation
from trustfed.incentives.credit import (
    DEFAULT_WEIGHTS,
    EVENT_CREDIT,
    EVENT_RECIPROCITY,
    KIND_EVALUATION_SERVED,
    CreditEvent,
    CreditLedger,
)
from trustfed.incentives.errors import (
    AttestationRejectedError,
    IdentifierError,
    IncentiveError,
    ReciprocityRefused,
    UnknownContributorError,
    UnknownCreditKindError,
)
from trustfed.incentives.identifiers import (
    LOCAL_PREFIX,
    Attribution,
    CitableIdentifier,
    ReleaseAttribution,
)
from trustfed.incentives.reciprocity import (
    EvaluationRequest,
    ReciprocityDecision,
    ReciprocityPolicy,
)
from trustfed.incentives.standing import (
    DEFAULT_TIERS,
    UNRANKED_TIER,
    ContributorStanding,
    StandingReport,
    StandingTier,
    tier_for,
)

__all__ = [
    "DEFAULT_TIERS",
    "DEFAULT_WEIGHTS",
    "UNRANKED_TIER",
    "EVENT_CREDIT",
    "EVENT_RECIPROCITY",
    "KIND_EVALUATION_SERVED",
    "LOCAL_PREFIX",
    "AttestationRejectedError",
    "Attribution",
    "CitableIdentifier",
    "CounterpartyRegistry",
    "ContributorStanding",
    "CreditEvent",
    "CreditLedger",
    "EvaluationRequest",
    "IdentifierError",
    "IncentiveError",
    "ReciprocityDecision",
    "ReciprocityPolicy",
    "ReciprocityRefused",
    "ReleaseAttribution",
    "ServiceAttestation",
    "StandingReport",
    "StandingTier",
    "UnknownContributorError",
    "UnknownCreditKindError",
    "tier_for",
]

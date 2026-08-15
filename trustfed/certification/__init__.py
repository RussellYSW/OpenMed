"""Component 4 -- multi-party certification and clinician-readable transfer.

A model becomes a *certified base* only when reviewers from more than one
institution sign off: :class:`ThresholdPolicy` counts approvals across distinct
institutions, and conflict-of-interest rules refuse a site that tries to certify
its own model. Every decision is appended to the ledger by
:class:`CertificationAuthority`, so certifications cannot be retroactively
edited, and disputes move the case through the state machine in
:mod:`trustfed.certification.states` (submitted, under_review, blocked,
remediated, certified, revoked).

:class:`FineTuningManual` is the fixed-schema, clinician-readable transfer
document a receiving site needs: intended use, data, preprocessing,
hyperparameters, failure modes, clinical caveats.

Certification here is a *governance* record. It is not regulatory clearance and
not a clinical safety assessment.

SCOPE NOTE, three parts, because the strength of every rule above depends on
them:

1. The conflict rules run against :class:`SubmissionFacts`. Open a case with the
   :class:`~trustfed.registry.bundle.ModelBundle` itself and the owner
   institution is derived from the bundle's ``published_by``; open it with a
   bundle id and a declared owner and the facts are the submitter's own claim,
   recorded as ``self_asserted``.
2. :meth:`CertificationAuthority.submit_signature` verifies a signature the
   authority did not produce, and :meth:`ReviewerKeyring.verify_only` gives it
   no private keys. :meth:`CertificationAuthority.review` signs *for* the
   reviewer with a key the authority holds -- convenient for demos and tests,
   and not a model of a reviewer signing remotely.
3. The dispute-side actions take an asserted actor unless a signed
   :class:`ActionAuthorization` -- a record bound to the specific action, and
   single-use -- is supplied; the decision log records which it was. A review
   signature is deliberately *not* accepted there: it is published in the log
   when it is filed, so it would be replayable by anyone who can read it.
"""

from __future__ import annotations

from trustfed.certification.actions import (
    AUTHORIZED_ACTIONS,
    ActionAuthorization,
)
from trustfed.certification.authority import (
    EVENT_CERTIFICATION,
    VALID_DECISIONS,
    CertificationAuthority,
    CertificationCase,
)
from trustfed.certification.disputes import BLOCKING_DECISION
from trustfed.certification.errors import (
    CertificationError,
    ConflictOfInterestError,
    InvalidTransitionError,
    KeyringReadOnlyError,
    ManualValidationError,
    SignatureRejectedError,
    ThresholdNotMetError,
    UnknownCaseError,
    UnknownReviewerError,
)
from trustfed.certification.keys import Reviewer, ReviewerKeyring, ReviewSignature
from trustfed.certification.manual import (
    MANUAL_SECTIONS,
    REQUIRED_FIELDS,
    FineTuningManual,
    ManualValidation,
    validate_manual,
)
from trustfed.certification.policy import (
    DEFAULT_CONFLICT_RULES,
    SubmissionFacts,
    ThresholdOutcome,
    ThresholdPolicy,
)
from trustfed.certification.states import (
    TRANSITIONS,
    CertificationState,
    can_transition,
    transitions_table,
)

__all__ = [
    "AUTHORIZED_ACTIONS",
    "BLOCKING_DECISION",
    "DEFAULT_CONFLICT_RULES",
    "EVENT_CERTIFICATION",
    "MANUAL_SECTIONS",
    "REQUIRED_FIELDS",
    "TRANSITIONS",
    "VALID_DECISIONS",
    "ActionAuthorization",
    "CertificationAuthority",
    "CertificationCase",
    "CertificationError",
    "CertificationState",
    "ConflictOfInterestError",
    "FineTuningManual",
    "InvalidTransitionError",
    "KeyringReadOnlyError",
    "ManualValidation",
    "ManualValidationError",
    "Reviewer",
    "ReviewSignature",
    "ReviewerKeyring",
    "SignatureRejectedError",
    "SubmissionFacts",
    "ThresholdNotMetError",
    "ThresholdOutcome",
    "ThresholdPolicy",
    "UnknownCaseError",
    "UnknownReviewerError",
    "can_transition",
    "transitions_table",
    "validate_manual",
]

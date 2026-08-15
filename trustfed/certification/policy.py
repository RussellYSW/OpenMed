"""Threshold-signature policy and conflict-of-interest rules.

A model becomes a *certified base* only when reviewers from more than one
institution sign off: k approving signatures spread across at least ``n``
distinct institutions. Counting signatures per institution rather than per
person is the point -- ten reviewers at the submitting hospital are still one
institution, and cannot certify their own model.

The threshold is a *policy* over signatures, not a cryptographic threshold
scheme (no key splitting, no Shamir shares). Each signature is bound to one
reviewer's registered key and to the institution registered with it, so a
signature cannot claim an institution its key is not bound to.

SCOPE NOTE: the rules are only as good as the facts they run against. See
:class:`SubmissionFacts` -- facts supplied by the submitting site are that
site's own claim about itself, and a site that misdeclares its owner
institution defeats every rule here. Use :meth:`SubmissionFacts.from_bundle` to
derive them from the published bundle instead. See also
:meth:`trustfed.certification.authority.CertificationAuthority.review` on where
the signing key actually lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from trustfed.certification.errors import CertificationError, ConflictOfInterestError
from trustfed.certification.keys import Reviewer, ReviewSignature

if TYPE_CHECKING:  # pragma: no cover - avoids a runtime import cycle
    from trustfed.registry.bundle import ModelBundle
    from trustfed.registry.registry import ModelRegistry

#: Conflict reason codes.
COI_SELF_CERTIFICATION = "self_certification"
COI_CONTRIBUTING_INSTITUTION = "contributing_institution"
COI_PERSONAL_CONTRIBUTION = "personal_contribution"
COI_DECLARED = "declared_conflict"

#: Threshold refusal reason codes.
REASON_INSUFFICIENT_APPROVALS = "insufficient_approvals"
REASON_INSUFFICIENT_INSTITUTIONS = "insufficient_institutions"
REASON_BLOCKING_REVIEW = "blocking_review"
REASON_MET = "threshold_met"


@dataclass(frozen=True)
class SubmissionFacts:
    """Who produced the model under review.

    Every conflict-of-interest rule reasons over these fields, so where they
    came from decides whether the rules are a control or a formality.

    ``self_asserted`` records that distinction. When it is True the facts are
    the submitting site's own claim about itself: a site that names some other
    institution as owner defeats
    :func:`rule_no_self_certification` outright, and the same lie defeats
    :func:`rule_no_contributing_institution` and :func:`rule_declared_conflict`.
    When it is False the facts were derived from the published bundle by
    :meth:`from_bundle`, so the owner institution is the one recorded in the
    content-addressed bundle rather than a free string.
    """

    bundle_id: str
    owner_institution: str
    contributor_institutions: Tuple[str, ...] = ()
    contributor_ids: Tuple[str, ...] = ()
    self_asserted: bool = True

    @classmethod
    def from_bundle(
        cls,
        bundle: "ModelBundle",
        *,
        registry: Optional["ModelRegistry"] = None,
    ) -> "SubmissionFacts":
        """Derive the facts from the bundle instead of trusting the submitter.

        ``owner_institution`` comes from ``bundle.published_by`` (falling back
        to ``bundle.attestation.client_id`` when the bundle has no publisher
        recorded), which is inside the content hash the bundle id commits to --
        changing it changes the bundle id. Contributor institutions and ids are
        collected from the bundle's own publisher and, when ``registry`` is
        given, from every registered ancestor's publisher and attested client.

        Parameters
        ----------
        bundle:
            A :class:`~trustfed.registry.bundle.ModelBundle`, or anything with
            the same ``bundle_id`` / ``published_by`` / ``attestation`` /
            ``parents`` attributes. Duck-typed so this module does not import
            the registry at runtime.
        registry:
            Optional :class:`~trustfed.registry.registry.ModelRegistry` used to
            resolve parent bundle ids to their publishers. Without it, parents
            contribute nothing to the conflict rules.

        Raises
        ------
        CertificationError
            If the object does not carry a bundle id, or records no publisher
            and no attested client id, so no owner can be derived.

        Notes
        -----
        This binds the certification layer to the registry's record of who
        published a model. It does not authenticate that record: with the mock
        attestor, ``published_by`` is still what the publishing process wrote.
        What it removes is the *separate*, unchecked assertion made at
        submission time, which could contradict the bundle without anything
        noticing.
        """
        bundle_id = str(getattr(bundle, "bundle_id", "") or "")
        if not bundle_id:
            raise CertificationError(
                "cannot derive submission facts: object has no bundle_id"
            )
        attestation = getattr(bundle, "attestation", None)
        attested_client = str(getattr(attestation, "client_id", "") or "")
        owner = str(getattr(bundle, "published_by", "") or "") or attested_client
        if not owner:
            raise CertificationError(
                f"cannot derive submission facts for {bundle_id}: the bundle "
                "records neither published_by nor an attested client id"
            )

        institutions: List[str] = [owner]
        contributors: List[str] = [attested_client] if attested_client else []
        for parent_id in tuple(getattr(bundle, "parents", ()) or ()):
            if registry is None or parent_id not in registry:
                continue
            parent = registry.get(parent_id)
            parent_owner = str(getattr(parent, "published_by", "") or "")
            if parent_owner:
                institutions.append(parent_owner)
            parent_client = str(
                getattr(getattr(parent, "attestation", None), "client_id", "") or ""
            )
            if parent_client:
                contributors.append(parent_client)

        return cls(
            bundle_id=bundle_id,
            owner_institution=owner,
            contributor_institutions=tuple(dict.fromkeys(institutions)),
            contributor_ids=tuple(dict.fromkeys(contributors)),
            self_asserted=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "bundle_id": self.bundle_id,
            "owner_institution": self.owner_institution,
            "contributor_institutions": list(self.contributor_institutions),
            "contributor_ids": list(self.contributor_ids),
            "self_asserted": self.self_asserted,
        }


#: A rule returns ``(reason_code, detail)`` when it objects, else ``None``.
ConflictRule = Callable[[Reviewer, SubmissionFacts], Optional[Tuple[str, str]]]


def rule_no_self_certification(
    reviewer: Reviewer, facts: SubmissionFacts
) -> Optional[Tuple[str, str]]:
    """Refuse a reviewer from the institution that owns the model."""
    if reviewer.institution == facts.owner_institution:
        return (
            COI_SELF_CERTIFICATION,
            f"{reviewer.reviewer_id} belongs to the submitting institution "
            f"{facts.owner_institution!r}",
        )
    return None


def rule_no_contributing_institution(
    reviewer: Reviewer, facts: SubmissionFacts
) -> Optional[Tuple[str, str]]:
    """Refuse a reviewer whose institution contributed to the model."""
    if reviewer.institution in facts.contributor_institutions:
        return (
            COI_CONTRIBUTING_INSTITUTION,
            f"{reviewer.institution!r} contributed training data or updates "
            "to this model",
        )
    return None


def rule_no_personal_contribution(
    reviewer: Reviewer, facts: SubmissionFacts
) -> Optional[Tuple[str, str]]:
    """Refuse a reviewer who is themselves listed as a contributor."""
    if reviewer.reviewer_id in facts.contributor_ids:
        return (
            COI_PERSONAL_CONTRIBUTION,
            f"{reviewer.reviewer_id} is a listed contributor to this model",
        )
    return None


def rule_declared_conflict(
    reviewer: Reviewer, facts: SubmissionFacts
) -> Optional[Tuple[str, str]]:
    """Refuse a reviewer who self-declared a conflict with this model or owner."""
    declared = set(reviewer.declared_conflicts)
    if facts.bundle_id in declared or facts.owner_institution in declared:
        return (
            COI_DECLARED,
            f"{reviewer.reviewer_id} declared a conflict covering this submission",
        )
    return None


#: Rules applied by default, in order.
DEFAULT_CONFLICT_RULES: Tuple[ConflictRule, ...] = (
    rule_no_self_certification,
    rule_no_contributing_institution,
    rule_no_personal_contribution,
    rule_declared_conflict,
)


@dataclass(frozen=True)
class ThresholdOutcome:
    """Whether the certification threshold is currently satisfied."""

    met: bool
    reason_code: str
    n_approvals: int = 0
    institutions: Tuple[str, ...] = ()
    blocking_reviewers: Tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "met": self.met,
            "reason_code": self.reason_code,
            "n_approvals": self.n_approvals,
            "institutions": list(self.institutions),
            "blocking_reviewers": list(self.blocking_reviewers),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ThresholdPolicy:
    """k-of-n approvals across distinct institutions, plus conflict rules.

    Parameters
    ----------
    k:
        Number of approving signatures required. When
        ``require_distinct_institutions`` is True, signatures from the same
        institution count once.
    min_institutions:
        Minimum number of distinct approving institutions. Defaults to 2, which
        is the "more than one institution" requirement.
    require_distinct_institutions:
        Count at most one approval per institution.
    conflict_rules:
        Rules consulted before a reviewer may sign at all.
    """

    k: int = 2
    min_institutions: int = 2
    require_distinct_institutions: bool = True
    conflict_rules: Tuple[ConflictRule, ...] = DEFAULT_CONFLICT_RULES

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.min_institutions < 1:
            raise ValueError("min_institutions must be >= 1")
        if self.require_distinct_institutions and self.min_institutions > self.k:
            raise ValueError(
                "min_institutions cannot exceed k when institutions are distinct"
            )

    def check_conflicts(
        self, reviewer: Reviewer, facts: SubmissionFacts
    ) -> Optional[Tuple[str, str]]:
        """Return the first conflict found for ``reviewer``, or ``None``."""
        for rule in self.conflict_rules:
            found = rule(reviewer, facts)
            if found is not None:
                return found
        return None

    def assert_may_review(self, reviewer: Reviewer, facts: SubmissionFacts) -> None:
        """Raise :class:`ConflictOfInterestError` if the reviewer is conflicted."""
        found = self.check_conflicts(reviewer, facts)
        if found is not None:
            raise ConflictOfInterestError(*found)

    def evaluate(
        self, signatures: Sequence[ReviewSignature]
    ) -> ThresholdOutcome:
        """Decide whether the given signatures satisfy the threshold.

        Assumes every signature was already verified against its reviewer's
        key; this method reasons about decisions and institutions only. A
        reviewer's most recent signature wins, so changing one's mind does not
        double-count.
        """
        latest: Dict[str, ReviewSignature] = {}
        for sig in signatures:
            latest[sig.reviewer_id] = sig

        blocking = tuple(
            sorted(s.reviewer_id for s in latest.values() if s.decision == "block")
        )
        if blocking:
            return ThresholdOutcome(
                met=False,
                reason_code=REASON_BLOCKING_REVIEW,
                n_approvals=sum(1 for s in latest.values() if s.decision == "approve"),
                institutions=(),
                blocking_reviewers=blocking,
                detail="at least one reviewer filed a blocking objection",
            )

        approvals = [s for s in latest.values() if s.decision == "approve"]
        institutions = tuple(sorted({s.institution for s in approvals}))
        counted = len(institutions) if self.require_distinct_institutions else len(approvals)

        if len(institutions) < self.min_institutions:
            return ThresholdOutcome(
                met=False,
                reason_code=REASON_INSUFFICIENT_INSTITUTIONS,
                n_approvals=counted,
                institutions=institutions,
                detail=(
                    f"approvals come from {len(institutions)} institution(s); "
                    f"policy requires {self.min_institutions}"
                ),
            )
        if counted < self.k:
            return ThresholdOutcome(
                met=False,
                reason_code=REASON_INSUFFICIENT_APPROVALS,
                n_approvals=counted,
                institutions=institutions,
                detail=f"{counted} counted approval(s); policy requires {self.k}",
            )
        return ThresholdOutcome(
            met=True,
            reason_code=REASON_MET,
            n_approvals=counted,
            institutions=institutions,
            detail=(
                f"{counted} approval(s) across {len(institutions)} institution(s)"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description of the policy."""
        return {
            "k": self.k,
            "min_institutions": self.min_institutions,
            "require_distinct_institutions": self.require_distinct_institutions,
            "conflict_rules": [r.__name__ for r in self.conflict_rules],
        }


__all__ = [
    "COI_CONTRIBUTING_INSTITUTION",
    "COI_DECLARED",
    "COI_PERSONAL_CONTRIBUTION",
    "COI_SELF_CERTIFICATION",
    "DEFAULT_CONFLICT_RULES",
    "REASON_BLOCKING_REVIEW",
    "REASON_INSUFFICIENT_APPROVALS",
    "REASON_INSUFFICIENT_INSTITUTIONS",
    "REASON_MET",
    "ConflictRule",
    "SubmissionFacts",
    "ThresholdOutcome",
    "ThresholdPolicy",
]

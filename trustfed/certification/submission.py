"""Opening a certification case, and where the facts about ownership come from.

Mixed into :class:`~trustfed.certification.authority.CertificationAuthority`.
Split out of that module because the question this code answers -- *who owns the
model under review, and who said so* -- is the one the conflict-of-interest
rules stand or fall on, and it deserves to be read on its own.

The short version: facts derived from a published bundle come from content the
bundle id commits to; facts supplied as a bundle id plus a free string are the
submitting site's own claim about itself, and are recorded as such.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from trustfed.certification.case import CertificationCase
from trustfed.certification.errors import CertificationError
from trustfed.certification.manual import FineTuningManual
from trustfed.certification.policy import SubmissionFacts
from trustfed.certification.states import CertificationState


class SubmissionMixin:
    """Case-opening half of the certification authority.

    Assumes the host class provides ``_cases``, ``_log``, ``_keyring`` and
    ``_require_manual``.
    """

    def submit(
        self,
        bundle: Any,
        *,
        owner_institution: Optional[str] = None,
        submitted_by: str,
        contributor_institutions: Sequence[str] = (),
        contributor_ids: Sequence[str] = (),
        manual: Optional[Mapping[str, Any]] = None,
        registry: Optional[Any] = None,
    ) -> CertificationCase:
        """Open a certification case.

        ``bundle`` may be any of three things, and which one you pass decides
        whether the conflict-of-interest rules are a control or a formality:

        * a :class:`~trustfed.registry.bundle.ModelBundle` (recommended) --
          the facts are derived with
          :meth:`~trustfed.certification.policy.SubmissionFacts.from_bundle`,
          so ``owner_institution`` comes from the bundle's own
          ``published_by`` field, which is inside the content hash its id
          commits to. Pass ``registry`` as well to pull contributing
          institutions from registered ancestors.
        * a :class:`~trustfed.certification.policy.SubmissionFacts` -- used
          as given.
        * a bundle-id string -- then ``owner_institution`` is required and the
          facts are **self-asserted**: they are the submitting site's own claim
          about who owns the model, checked against nothing. A site that names
          another institution as owner passes
          :func:`~trustfed.certification.policy.rule_no_self_certification`
          while its own reviewers approve. Submissions made this way are logged
          with ``"self_asserted": true`` so a reader of the decision log can
          tell which submissions rest on an unchecked claim.

        Raises
        ------
        CertificationError
            If a case already exists for this bundle, if a bundle id was given
            without ``owner_institution``, if a bundle object was given with a
            contradicting ``owner_institution``, or if the manual is required
            and missing/invalid.
        """
        facts = self._facts_for(
            bundle,
            owner_institution=owner_institution,
            contributor_institutions=contributor_institutions,
            contributor_ids=contributor_ids,
            registry=registry,
        )
        bundle_id = facts.bundle_id
        if bundle_id in self._cases:
            raise CertificationError(
                f"a certification case already exists for {bundle_id} "
                f"(state {self._cases[bundle_id].state.value})"
            )
        manual_ok: Optional[bool] = None
        if manual is not None:
            validation = FineTuningManual.from_dict(manual).validate()
            manual_ok = validation.ok
            if self._require_manual:
                validation.raise_if_invalid()
        elif self._require_manual:
            raise CertificationError(
                "this authority requires a fine-tuning manual at submission"
            )
        case = CertificationCase(
            bundle_id=bundle_id,
            facts=facts,
            state=CertificationState.SUBMITTED,
            submitted_by=submitted_by,
            manual_ok=manual_ok,
            # The keyring holds the revocation state, so the case's own view of
            # who approves has to consult it; otherwise a published snapshot
            # would credit an institution the threshold no longer counts.
            verify=self._keyring.verify,
        )
        self._cases[bundle_id] = case
        self._log(
            bundle_id,
            "submitted",
            submitted_by,
            to_state=CertificationState.SUBMITTED,
            detail=(
                f"owner institution {facts.owner_institution}"
                + ("" if not facts.self_asserted else " (self-asserted)")
            ),
            extra={
                "facts": facts.to_dict(),
                "manual_ok": manual_ok,
                "self_asserted": facts.self_asserted,
            },
        )
        return case

    def _facts_for(
        self,
        bundle: Any,
        *,
        owner_institution: Optional[str],
        contributor_institutions: Sequence[str],
        contributor_ids: Sequence[str],
        registry: Optional[Any],
    ) -> SubmissionFacts:
        """Build the :class:`SubmissionFacts` for a call to :meth:`submit`."""
        if isinstance(bundle, SubmissionFacts):
            return bundle
        if isinstance(bundle, str):
            if not owner_institution:
                raise CertificationError(
                    "submitting by bundle id requires owner_institution; pass the "
                    "ModelBundle itself to have the owner derived from it instead"
                )
            return SubmissionFacts(
                bundle_id=bundle,
                owner_institution=owner_institution,
                contributor_institutions=tuple(contributor_institutions),
                contributor_ids=tuple(contributor_ids),
                self_asserted=True,
            )
        derived = SubmissionFacts.from_bundle(bundle, registry=registry)
        if owner_institution and owner_institution != derived.owner_institution:
            raise CertificationError(
                f"declared owner {owner_institution!r} contradicts the bundle's "
                f"published_by {derived.owner_institution!r}; the bundle wins"
            )
        extra_institutions = tuple(
            i for i in contributor_institutions if i not in derived.contributor_institutions
        )
        extra_ids = tuple(i for i in contributor_ids if i not in derived.contributor_ids)
        if not extra_institutions and not extra_ids:
            return derived
        # Declared contributors *widen* the conflict set; they can never narrow
        # it, so accepting them does not weaken the derived facts.
        return SubmissionFacts(
            bundle_id=derived.bundle_id,
            owner_institution=derived.owner_institution,
            contributor_institutions=derived.contributor_institutions + extra_institutions,
            contributor_ids=derived.contributor_ids + extra_ids,
            self_asserted=False,
        )


__all__ = ["SubmissionMixin"]

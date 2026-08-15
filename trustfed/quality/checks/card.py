"""Model-card completeness: is the submission documented well enough to review?

This is the cheapest check to run and the one a reviewer needs first: a model
nobody can describe cannot be reviewed, reproduced or safely reused. It is a
string-and-structure check only -- it verifies that a section is *present and
non-trivial*, never that its contents are true. The two structural checks that
used to live here (bundle schema, lineage/attestation presence) are in
``trustfed.quality.checks.provenance``.

Two model-card vocabularies are in use in this repository and a submitter must
not be forced to satisfy contradictory gates, so the completeness check accepts
either and scores the card under the better-fitting one:

* :data:`MITCHELL_CARD_SECTIONS` -- the nine sections of Mitchell et al.,
  "Model Cards for Model Reporting" (FAT* 2019), which is the schema the
  registry enforces (``trustfed.registry.model_card.CARD_SECTIONS``); the tuple
  below is a literal copy, because this package does not import the registry.
* :data:`TRIMMED_CARD_SECTIONS` -- a flatter vocabulary that names
  ``out_of_scope_use``, ``limitations`` and ``contact`` as top-level sections.

The two are reconciled by :data:`SECTION_ALIASES`, which also locates the
trimmed names where a Mitchell card puts them (``intended_use.out_of_scope``,
``caveats_and_recommendations.limitations``, ``model_details.contact``). A card
that passes ``ModelCard.validate()`` in the registry therefore passes here too.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from trustfed.quality.bundle import BundleView, QualityEvidence, as_mapping
from trustfed.quality.checks.base import Check
from trustfed.quality.errors import CheckConfigError
from trustfed.quality.report import CheckResult, Status

#: The nine sections of Mitchell et al. (2019), matching the registry's schema.
#: Copied as a literal rather than imported: the quality analyzer must run
#: against a bundle produced by a different version of the registry.
MITCHELL_CARD_SECTIONS: Tuple[str, ...] = (
    "model_details",
    "intended_use",
    "factors",
    "metrics",
    "evaluation_data",
    "training_data",
    "quantitative_analyses",
    "ethical_considerations",
    "caveats_and_recommendations",
)

#: The flatter vocabulary, which promotes three of the Mitchell sub-topics to
#: top-level sections. Kept because it is what a submitter writing a card by
#: hand tends to produce.
TRIMMED_CARD_SECTIONS: Tuple[str, ...] = (
    "model_details",
    "intended_use",
    "out_of_scope_use",
    "training_data",
    "evaluation_data",
    "metrics",
    "limitations",
    "ethical_considerations",
    "contact",
)

#: Vocabularies the completeness check will score a card against; the card is
#: graded under whichever it fits best, and the report names which was used.
ACCEPTED_CARD_VOCABULARIES: Dict[str, Tuple[str, ...]] = {
    "mitchell": MITCHELL_CARD_SECTIONS,
    "trimmed": TRIMMED_CARD_SECTIONS,
}

#: Default required set. Both vocabularies are accepted; this names the one the
#: registry enforces, so "required" here and "required" there agree.
REQUIRED_CARD_SECTIONS: Tuple[str, ...] = MITCHELL_CARD_SECTIONS

#: Where a section may live when it is not a top-level key: ``(section, key)``
#: pairs are looked up inside another section, bare strings are synonyms.
SECTION_ALIASES: Dict[str, Tuple[Any, ...]] = {
    "out_of_scope_use": (
        ("intended_use", "out_of_scope_use"),
        ("intended_use", "out_of_scope"),
        ("intended_use", "not_intended_for"),
    ),
    "limitations": (
        ("caveats_and_recommendations", "limitations"),
        ("caveats_and_recommendations", "caveats"),
        ("quantitative_analyses", "limitations"),
    ),
    "contact": (
        ("model_details", "contact"),
        ("model_details", "contact_email"),
        ("model_details", "owner"),
    ),
    "caveats_and_recommendations": ("caveats", "limitations"),
    "factors": ("subgroups",),
    "quantitative_analyses": ("subgroup_metrics", "quantitative_analysis"),
}

#: Sections that are strongly recommended but not required.
OPTIONAL_CARD_SECTIONS: Tuple[str, ...] = ("license", "citation", "caveats")

MIN_SECTION_CHARS = 10


def _is_filled(value: Any, min_chars: int = MIN_SECTION_CHARS) -> bool:
    """Whether a card section counts as populated rather than a stub."""
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) >= min_chars
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def section_value(card: Mapping[str, Any], name: str) -> Any:
    """Return the card's value for ``name``, following :data:`SECTION_ALIASES`.

    Looks for a top-level key first, then for synonyms and for the nested
    location the other vocabulary uses. Returns ``None`` when the section is
    absent under every known name.
    """
    value = card.get(name)
    if value is not None:
        return value
    for alias in SECTION_ALIASES.get(name, ()):
        if isinstance(alias, tuple):
            parent, key = alias
            nested = as_mapping(card.get(parent)).get(key)
            if nested is not None:
                return nested
        else:
            value = card.get(alias)
            if value is not None:
                return value
    return None


class ModelCardCompletenessCheck(Check):
    """Are the required model-card sections present and non-trivial?

    Parameters
    ----------
    required:
        Section names that must be present. When given, only this set is used.
        When ``None`` (the default) the card is scored against every vocabulary
        in ``vocabularies`` and graded under the best-fitting one, so the
        registry's schema and the flatter hand-written one both pass.
    vocabularies:
        Named required-section sets; defaults to
        :data:`ACCEPTED_CARD_VOCABULARIES`.
    warn_below, fail_below:
        Completeness fractions below which the result is ``WARN`` / ``FAIL``.
    min_chars:
        Minimum length for a free-text section to count as filled; guards
        against placeholder values like ``"TODO"``.

    This checks presence and length only. It cannot tell whether a stated
    intended use is accurate -- that is what human review is for.
    """

    check_id = "model_card_completeness"
    title = "Model card completeness"

    def __init__(
        self,
        required: Optional[Sequence[str]] = None,
        *,
        vocabularies: Optional[Mapping[str, Sequence[str]]] = None,
        warn_below: float = 1.0,
        fail_below: float = 0.6,
        min_chars: int = MIN_SECTION_CHARS,
    ):
        if not 0.0 <= fail_below <= warn_below <= 1.0:
            raise CheckConfigError(
                "thresholds must satisfy 0 <= fail_below <= warn_below <= 1"
            )
        if required is not None:
            self.vocabularies: Dict[str, Tuple[str, ...]] = {
                "custom": tuple(required)
            }
        else:
            source = vocabularies or ACCEPTED_CARD_VOCABULARIES
            self.vocabularies = {k: tuple(v) for k, v in source.items()}
        if not self.vocabularies or not any(self.vocabularies.values()):
            raise CheckConfigError("at least one non-empty vocabulary is required")
        self.warn_below = float(warn_below)
        self.fail_below = float(fail_below)
        self.min_chars = int(min_chars)

    @property
    def required(self) -> Tuple[str, ...]:
        """The first accepted vocabulary, for callers that want one list."""
        return next(iter(self.vocabularies.values()))

    def _score(
        self, card: Dict[str, Any], sections: Sequence[str]
    ) -> Tuple[float, List[str], List[str]]:
        """Return ``(fraction, present, missing)`` for one vocabulary."""
        present = [
            s for s in sections if _is_filled(section_value(card, s), self.min_chars)
        ]
        missing = [s for s in sections if s not in present]
        fraction = len(present) / float(len(sections)) if sections else 1.0
        return fraction, present, missing

    def run(self, bundle: BundleView, evidence: QualityEvidence) -> CheckResult:
        """Score the card and report which sections are missing."""
        card = bundle.model_card
        if not card:
            return self._result(
                Status.FAIL,
                "No model card found on the bundle.",
                details={"accepted_vocabularies": {
                    k: list(v) for k, v in self.vocabularies.items()
                }},
                recommendation=(
                    "Attach a model card documenting intended use, training "
                    "data, evaluation and limitations."
                ),
            )
        scored = {
            name: self._score(card, sections)
            for name, sections in self.vocabularies.items()
        }
        vocabulary = max(scored, key=lambda k: scored[k][0])
        fraction, present, missing = scored[vocabulary]
        optional_present = [
            s for s in OPTIONAL_CARD_SECTIONS if _is_filled(card.get(s), self.min_chars)
        ]
        details: Dict[str, Any] = {
            "vocabulary": vocabulary,
            "required_sections": list(self.vocabularies[vocabulary]),
            "completeness": round(fraction, 4),
            "completeness_by_vocabulary": {
                k: round(v[0], 4) for k, v in sorted(scored.items())
            },
            "present_sections": present,
            "missing_sections": missing,
            "optional_sections_present": optional_present,
        }
        if fraction < self.fail_below:
            return self._result(
                Status.FAIL,
                f"Model card is {fraction:.0%} complete against the "
                f"{vocabulary} section vocabulary; "
                f"{len(missing)} required section(s) missing.",
                details=details,
                recommendation=f"Add the missing sections: {', '.join(missing)}.",
            )
        if fraction < self.warn_below:
            return self._result(
                Status.WARN,
                f"Model card is {fraction:.0%} complete against the "
                f"{vocabulary} section vocabulary; missing "
                f"{', '.join(missing)}.",
                details=details,
                recommendation=f"Add the missing sections: {', '.join(missing)}.",
            )
        return self._result(
            Status.PASS,
            "All required model-card sections are present "
            f"({vocabulary} section vocabulary).",
            details=details,
        )


__all__ = [
    "ModelCardCompletenessCheck",
    "ACCEPTED_CARD_VOCABULARIES",
    "MITCHELL_CARD_SECTIONS",
    "TRIMMED_CARD_SECTIONS",
    "REQUIRED_CARD_SECTIONS",
    "OPTIONAL_CARD_SECTIONS",
    "SECTION_ALIASES",
    "section_value",
]

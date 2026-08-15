"""Model cards, following the section structure of Mitchell et al. (2019).

Reference: Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy
Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji and Timnit
Gebru, "Model Cards for Model Reporting", FAT* 2019, arXiv:1810.03993.

The nine sections of that paper are represented verbatim as fields so a reader
can map a bundle onto the published schema. Validation here is *structural*: it
checks that a section is present and non-empty. It cannot check that the
content is accurate, and it is not a clinical or regulatory review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trustfed.registry.errors import ValidationError

#: The nine sections defined by Mitchell et al. (2019).
CARD_SECTIONS: Tuple[str, ...] = (
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

#: Keys expected inside ``model_details`` for a card to count as complete.
REQUIRED_MODEL_DETAILS: Tuple[str, ...] = (
    "name",
    "version",
    "owner",
    "date",
    "model_type",
    "license",
)


@dataclass(frozen=True)
class ModelCardValidation:
    """Structural verdict for a model card."""

    ok: bool
    missing_sections: Tuple[str, ...] = ()
    missing_details: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the verdict."""
        return {
            "ok": self.ok,
            "missing_sections": list(self.missing_sections),
            "missing_details": list(self.missing_details),
            "warnings": list(self.warnings),
        }

    def raise_if_invalid(self) -> None:
        """Raise :class:`ValidationError` when the card is incomplete."""
        if not self.ok:
            raise ValidationError(
                "model card incomplete: missing sections "
                f"{list(self.missing_sections)}, missing model_details "
                f"{list(self.missing_details)}"
            )


@dataclass(frozen=True)
class ModelCard:
    """A model card with the nine Mitchell et al. sections.

    Every section is a free-form mapping so that domain-specific detail can be
    recorded without schema churn; :meth:`validate` enforces presence, not
    shape. ``model_details`` additionally needs the keys in
    :data:`REQUIRED_MODEL_DETAILS`.
    """

    model_details: Mapping[str, Any] = field(default_factory=dict)
    intended_use: Mapping[str, Any] = field(default_factory=dict)
    factors: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evaluation_data: Mapping[str, Any] = field(default_factory=dict)
    training_data: Mapping[str, Any] = field(default_factory=dict)
    quantitative_analyses: Mapping[str, Any] = field(default_factory=dict)
    ethical_considerations: Mapping[str, Any] = field(default_factory=dict)
    caveats_and_recommendations: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> ModelCardValidation:
        """Check that every section and required detail key is present.

        Warnings (which do not fail validation) flag sections that reviewers
        most often leave thin: subgroup factors and ethical considerations.
        """
        missing_sections = tuple(
            name for name in CARD_SECTIONS if not getattr(self, name)
        )
        details = self.model_details or {}
        missing_details = tuple(
            key
            for key in REQUIRED_MODEL_DETAILS
            if not str(details.get(key, "")).strip()
        )
        warnings: List[str] = []
        if not (self.factors or {}).get("groups"):
            warnings.append("factors.groups is empty: no subgroups declared")
        if not (self.caveats_and_recommendations or {}).get("caveats"):
            warnings.append("no caveats recorded")
        return ModelCardValidation(
            ok=not missing_sections and not missing_details,
            missing_sections=missing_sections,
            missing_details=missing_details,
            warnings=tuple(warnings),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the card."""
        return {name: dict(getattr(self, name) or {}) for name in CARD_SECTIONS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelCard":
        """Rebuild a card from :meth:`to_dict` output.

        Unknown keys are rejected rather than silently dropped.
        """
        unknown = sorted(set(data) - set(CARD_SECTIONS))
        if unknown:
            raise ValidationError(f"unknown model-card sections: {unknown}")
        return cls(**{name: dict(data.get(name, {})) for name in CARD_SECTIONS})

    def to_markdown(self) -> str:
        """Render the card as Markdown for human review."""
        lines: List[str] = ["# Model card"]
        for name in CARD_SECTIONS:
            lines.append("")
            lines.append(f"## {name.replace('_', ' ').capitalize()}")
            section = getattr(self, name) or {}
            if not section:
                lines.append("_not provided_")
                continue
            for key, value in sorted(section.items()):
                lines.append(f"- **{key}**: {_render(value)}")
        return "\n".join(lines) + "\n"


def _render(value: Any) -> str:
    """Render one model-card value for Markdown output."""
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={v}" for k, v in sorted(value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(str(v) for v in value)
    return str(value)


def as_dict(card: ModelCard) -> Dict[str, Any]:
    """Return ``asdict`` of a card (kept for symmetry with dataclass users)."""
    return asdict(card)


__all__ = [
    "CARD_SECTIONS",
    "REQUIRED_MODEL_DETAILS",
    "ModelCard",
    "ModelCardValidation",
    "as_dict",
]

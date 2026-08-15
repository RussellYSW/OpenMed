"""The clinician-readable fine-tuning manual: fixed schema plus validator.

A model that arrives at a new hospital is only reusable if someone there can
answer six questions without reading the training code: what is it for, what was
it trained on, how were inputs prepared, how was it fitted, how does it fail, and
what must a clinician watch for. This module fixes those six sections as a
schema, requires named fields inside each, and renders the result as Markdown.

Validation is structural: it checks that the author answered each question, not
that the answer is correct. It is not a regulatory submission, not a substitute
for local clinical validation, and confers no approval of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trustfed.certification.errors import ManualValidationError

#: The six required sections, in the order they are rendered.
MANUAL_SECTIONS: Tuple[str, ...] = (
    "intended_use",
    "data",
    "preprocessing",
    "hyperparameters",
    "failure_modes",
    "clinical_caveats",
)

#: Fields each section must define for the manual to be usable at a new site.
REQUIRED_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "intended_use": ("task", "population", "care_setting", "not_intended_for"),
    "data": ("sources", "n_records", "inclusion_criteria", "label_definition"),
    "preprocessing": ("steps", "normalization", "missing_data"),
    "hyperparameters": ("optimizer", "learning_rate", "epochs", "batch_size"),
    "failure_modes": ("known_failure_modes", "monitoring"),
    "clinical_caveats": ("human_oversight", "contraindications", "escalation"),
}


@dataclass(frozen=True)
class ManualValidation:
    """Structural verdict for a fine-tuning manual."""

    ok: bool
    missing_sections: Tuple[str, ...] = ()
    missing_fields: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the verdict."""
        return {
            "ok": self.ok,
            "missing_sections": list(self.missing_sections),
            "missing_fields": list(self.missing_fields),
            "warnings": list(self.warnings),
        }

    def raise_if_invalid(self) -> None:
        """Raise :class:`ManualValidationError` when the manual is incomplete."""
        if not self.ok:
            raise ManualValidationError(
                "fine-tuning manual incomplete: missing sections "
                f"{list(self.missing_sections)}, missing fields "
                f"{list(self.missing_fields)}"
            )


@dataclass(frozen=True)
class FineTuningManual:
    """A fixed-schema transfer manual written for the receiving clinical team.

    Each section is a mapping so sites can add local detail, but the fields in
    :data:`REQUIRED_FIELDS` must be present and non-empty for
    :meth:`validate` to pass.
    """

    intended_use: Mapping[str, Any] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    failure_modes: Mapping[str, Any] = field(default_factory=dict)
    clinical_caveats: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> ManualValidation:
        """Check every required section and field is present and non-empty."""
        missing_sections: List[str] = []
        missing_fields: List[str] = []
        warnings: List[str] = []
        for section in MANUAL_SECTIONS:
            content = getattr(self, section) or {}
            if not content:
                missing_sections.append(section)
                continue
            for key in REQUIRED_FIELDS[section]:
                value = content.get(key)
                if value is None or (
                    isinstance(value, (str, list, tuple, dict)) and len(value) == 0
                ):
                    missing_fields.append(f"{section}.{key}")
        if not (self.failure_modes or {}).get("known_failure_modes"):
            warnings.append(
                "no known failure modes recorded: reviewers usually treat this as "
                "under-reporting rather than absence of failures"
            )
        if not (self.clinical_caveats or {}).get("human_oversight"):
            warnings.append("no human-oversight requirement stated")
        return ManualValidation(
            ok=not missing_sections and not missing_fields,
            missing_sections=tuple(missing_sections),
            missing_fields=tuple(missing_fields),
            warnings=tuple(warnings),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of the manual."""
        return {name: dict(getattr(self, name) or {}) for name in MANUAL_SECTIONS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FineTuningManual":
        """Rebuild a manual from :meth:`to_dict` output.

        Raises
        ------
        ManualValidationError
            If the mapping carries sections outside the fixed schema.
        """
        unknown = sorted(set(data) - set(MANUAL_SECTIONS))
        if unknown:
            raise ManualValidationError(f"unknown manual sections: {unknown}")
        return cls(**{name: dict(data.get(name, {})) for name in MANUAL_SECTIONS})

    def to_markdown(self) -> str:
        """Render the manual as Markdown for the receiving clinical team."""
        titles = {
            "intended_use": "What this model is for",
            "data": "What it was trained on",
            "preprocessing": "How inputs must be prepared",
            "hyperparameters": "How it was fitted",
            "failure_modes": "How it fails",
            "clinical_caveats": "Clinical caveats",
        }
        lines: List[str] = ["# Fine-tuning manual"]
        for section in MANUAL_SECTIONS:
            lines.append("")
            lines.append(f"## {titles[section]}")
            content = getattr(self, section) or {}
            if not content:
                lines.append("_not provided_")
                continue
            for key, value in sorted(content.items()):
                lines.append(f"- **{key.replace('_', ' ')}**: {_render(value)}")
        return "\n".join(lines) + "\n"


def _render(value: Any) -> str:
    """Render one manual value for Markdown output."""
    if isinstance(value, Mapping):
        return "; ".join(f"{k}: {v}" for k, v in sorted(value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "; ".join(str(v) for v in value)
    return str(value)


def validate_manual(data: Mapping[str, Any]) -> ManualValidation:
    """Validate a manual given as a plain mapping (e.g. from a bundle)."""
    return FineTuningManual.from_dict(data).validate()


__all__ = [
    "MANUAL_SECTIONS",
    "REQUIRED_FIELDS",
    "FineTuningManual",
    "ManualValidation",
    "validate_manual",
]

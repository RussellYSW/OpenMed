"""The individual quality checks, and the default pipeline order.

Each check is independent and can be used, replaced or subclassed on its own;
:func:`default_checks` returns the standard set in the order a reviewer reads
them (documentation, then structure, then measured behaviour).
"""

from __future__ import annotations

from typing import List

from trustfed.quality.checks.base import Check
from trustfed.quality.checks.card import (
    ACCEPTED_CARD_VOCABULARIES,
    MITCHELL_CARD_SECTIONS,
    OPTIONAL_CARD_SECTIONS,
    REQUIRED_CARD_SECTIONS,
    TRIMMED_CARD_SECTIONS,
    ModelCardCompletenessCheck,
)
from trustfed.quality.checks.performance import (
    REQUIRED_METRICS,
    MetricsPresentCheck,
    SubgroupGapCheck,
)
from trustfed.quality.checks.privacy import (
    MembershipInferenceCheck,
    ParameterNormCheck,
    threshold_attack_advantage,
)
from trustfed.quality.checks.provenance import (
    LineageAttestationCheck,
    MetadataSchemaCheck,
)


def default_checks() -> List[Check]:
    """Return a fresh list of the standard checks, in reporting order."""
    return [
        ModelCardCompletenessCheck(),
        MetadataSchemaCheck(),
        LineageAttestationCheck(),
        MetricsPresentCheck(),
        SubgroupGapCheck(),
        MembershipInferenceCheck(),
        ParameterNormCheck(),
    ]


__all__ = [
    "Check",
    "ModelCardCompletenessCheck",
    "MetadataSchemaCheck",
    "LineageAttestationCheck",
    "MetricsPresentCheck",
    "SubgroupGapCheck",
    "MembershipInferenceCheck",
    "ParameterNormCheck",
    "threshold_attack_advantage",
    "default_checks",
    "ACCEPTED_CARD_VOCABULARIES",
    "MITCHELL_CARD_SECTIONS",
    "TRIMMED_CARD_SECTIONS",
    "REQUIRED_CARD_SECTIONS",
    "OPTIONAL_CARD_SECTIONS",
    "REQUIRED_METRICS",
]

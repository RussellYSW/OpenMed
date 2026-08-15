"""Automated quality analysis of a submitted model bundle (Component 5).

A :class:`~trustfed.quality.analyzer.QualityAnalyzer` runs independent checks --
model-card completeness, bundle schema, lineage/attestation presence, metric
sanity, subgroup performance gaps, a membership-inference leakage screen, and a
parameter-norm outlier screen -- and emits a structured
:class:`~trustfed.quality.report.QualityReport` as JSON and Markdown.

Everything runs locally with numpy only. A local LLM can draft the narrative if
one is configured (:class:`~trustfed.quality.summarize.LocalLLMSummarizer`), but
the default :class:`~trustfed.quality.summarize.TemplateSummarizer` is
deterministic and needs no model, so the analysis works with nothing installed
beyond this package.

The report is a triage aid for human reviewers. Passing every check does not
mean a model is safe, generalizable or clinically validated.
"""

from __future__ import annotations

from trustfed.quality.analyzer import ANALYZER_VERSION, QualityAnalyzer, analyze_bundle
from trustfed.quality.bundle import BUNDLE_FIELDS, BundleView, QualityEvidence
from trustfed.quality.checks import (
    Check,
    LineageAttestationCheck,
    MembershipInferenceCheck,
    MetadataSchemaCheck,
    MetricsPresentCheck,
    ModelCardCompletenessCheck,
    ParameterNormCheck,
    SubgroupGapCheck,
    default_checks,
    threshold_attack_advantage,
)
from trustfed.quality.errors import BundleFormatError, CheckConfigError, QualityError
from trustfed.quality.report import DISCLAIMER, CheckResult, QualityReport, Status
from trustfed.quality.summarize import (
    LLMSummarizer,
    LocalLLMSummarizer,
    TemplateSummarizer,
)

__all__ = [
    "QualityAnalyzer",
    "analyze_bundle",
    "ANALYZER_VERSION",
    "BundleView",
    "QualityEvidence",
    "BUNDLE_FIELDS",
    "Check",
    "default_checks",
    "ModelCardCompletenessCheck",
    "MetadataSchemaCheck",
    "LineageAttestationCheck",
    "MetricsPresentCheck",
    "SubgroupGapCheck",
    "MembershipInferenceCheck",
    "ParameterNormCheck",
    "threshold_attack_advantage",
    "QualityReport",
    "CheckResult",
    "Status",
    "DISCLAIMER",
    "LLMSummarizer",
    "TemplateSummarizer",
    "LocalLLMSummarizer",
    "QualityError",
    "BundleFormatError",
    "CheckConfigError",
]

"""Component 3 -- the model registry: bundles, model cards, lineage graph.

A *bundle* is a versioned release: a reference to weights, a model card in the
Mitchell et al. (2019) format, the attestation of the pipeline that produced it,
an evaluation report, and an optional clinician-readable fine-tuning manual. The
bundle id is the hash of that content, so the id is itself an integrity check.

Publishing a derived model links it to its parents, so the registry accumulates
a derivation **graph**; :meth:`ModelRegistry.verify_lineage` walks that graph to
its roots and returns a structured verdict. Every publish is appended to the
tamper-evident ledger in :mod:`trustfed.ledger`.

The registry records claims and their provenance. It does not validate clinical
performance, does not fetch or check weights at the recorded URI, and inherits
every limitation of the attestation backend that signed the pipeline quote.
"""

from __future__ import annotations

from trustfed.registry.bundle import (
    BUNDLE_ID_PREFIX,
    EvaluationReport,
    ModelBundle,
    PipelineAttestation,
    WeightsRef,
)
from trustfed.registry.errors import (
    AttestationRequiredError,
    BundleNotFoundError,
    DuplicateBundleError,
    LineageError,
    RegistryError,
    ValidationError,
)
from trustfed.registry.lineage import LineageGraph, LineageIssue, LineageVerdict
from trustfed.registry.model_card import (
    CARD_SECTIONS,
    REQUIRED_MODEL_DETAILS,
    ModelCard,
    ModelCardValidation,
)
from trustfed.registry.registry import EVENT_MODEL_PUBLISHED, ModelRegistry

__all__ = [
    "BUNDLE_ID_PREFIX",
    "CARD_SECTIONS",
    "EVENT_MODEL_PUBLISHED",
    "REQUIRED_MODEL_DETAILS",
    "AttestationRequiredError",
    "BundleNotFoundError",
    "DuplicateBundleError",
    "EvaluationReport",
    "LineageError",
    "LineageGraph",
    "LineageIssue",
    "LineageVerdict",
    "ModelBundle",
    "ModelCard",
    "ModelCardValidation",
    "ModelRegistry",
    "PipelineAttestation",
    "RegistryError",
    "ValidationError",
    "WeightsRef",
]

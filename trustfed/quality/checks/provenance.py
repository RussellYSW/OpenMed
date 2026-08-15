"""Structural checks: bundle schema validity and provenance presence.

Two checks that ask not "is this model good?" but "is this submission put
together the way the commons expects?" -- the top-level bundle fields, and
whether there is provenance (lineage plus an attestation record) to verify at
all. Neither verifies a signature: cryptographic verification belongs to the
ledger and attestation components, which own the keys.

Both deliberately avoid importing ``trustfed.registry``, so a quality review
still runs against a bundle produced by a different version of it. What they do
share with the registry is vocabulary, which is where the subtlety lives: an
empty ``parents`` tuple is the registry's way of saying "trained
independently", not "provenance withheld", and both checks treat it as a root
release rather than a defect.
"""

from __future__ import annotations

from typing import Any, Dict, List

from trustfed.quality.bundle import BundleView, QualityEvidence, as_mapping
from trustfed.quality.checks.base import Check
from trustfed.quality.errors import CheckConfigError
from trustfed.quality.report import CheckResult, Status


class MetadataSchemaCheck(Check):
    """Is the bundle structurally what the registry expects?

    Verifies that the five bundle fields are present and of a plausible shape
    (mappings for the card/metrics/attestation, a sequence for lineage, a path
    for the weights). It does not validate the registry's schema version and
    does not import the registry package -- deliberately, so that a quality
    review still runs against a bundle produced by a different version.

    An empty-but-declared lineage field is *not* counted as missing: the
    registry gives a root release an empty ``parents`` tuple, and a root release
    is a normal submission, not a malformed one.
    """

    check_id = "metadata_schema"
    title = "Bundle metadata and schema"

    def __init__(self, *, require_weights_path: bool = True):
        self.require_weights_path = bool(require_weights_path)

    def run(self, bundle: BundleView, evidence: QualityEvidence) -> CheckResult:
        """Report missing or malformed top-level bundle fields."""
        missing = bundle.missing_fields()
        if not self.require_weights_path and "weights_path" in missing:
            missing = [m for m in missing if m != "weights_path"]
        root_submission = "lineage" in missing and bundle.declares_lineage_field
        if root_submission:
            missing = [m for m in missing if m != "lineage"]
        problems: List[str] = []
        if bundle.model_card and not isinstance(bundle.model_card, dict):
            problems.append("model_card is not a mapping")
        raw_metrics = bundle.metrics
        for key, value in raw_metrics.items():
            if key == "subgroups":
                continue
            if isinstance(value, (dict, list, tuple, str, bool)) or value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                problems.append(f"metric '{key}' is not numeric")
        details: Dict[str, Any] = {
            "present_fields": bundle.present_fields(),
            "missing_fields": missing,
            "problems": problems,
            "root_submission": root_submission,
        }
        if missing or problems:
            status = Status.FAIL if len(missing) >= 3 or problems else Status.WARN
            return self._result(
                status,
                "Bundle is missing or malformed fields: "
                + ", ".join(missing + problems),
                details=details,
                recommendation=(
                    "Populate every bundle field before submission; the registry "
                    "and the reviewers both key off them."
                ),
            )
        return self._result(
            Status.PASS,
            "All expected bundle fields are present and well formed.",
            details=details,
        )


class LineageAttestationCheck(Check):
    """Does the submission carry provenance and an attestation record?

    Checks presence and minimal structure of the lineage entries and the
    attestation record. It does **not** verify signatures: signature
    verification belongs to the ledger and attestation components, which own the
    keys. A ``PASS`` here means "there is something to verify", not "verified".

    Root releases
    -------------
    A model trained independently has no parents, so the registry gives it an
    empty ``parents`` tuple. That is the shape of every first release and of
    every non-derived submission, and it is *not* absent provenance: the
    attestation record is what says where the model came from. An empty lineage
    is therefore only a finding when there is no attestation either --

    ===================  ==============  ===========================
    lineage              attestation     result
    ===================  ==============  ===========================
    entries present      present         ``PASS``
    empty (declared)     present         ``PASS`` (root submission)
    empty (no field)     present         ``WARN`` (root assumed)
    any                  absent          ``FAIL``
    ===================  ==============  ===========================

    ``min_lineage_entries`` still applies to bundles that do carry an
    attestation-free lineage, and raising it above 1 tightens what counts as
    "has parents" without making a root release fail.
    """

    check_id = "lineage_attestation"
    title = "Lineage and attestation presence"

    def __init__(self, *, min_lineage_entries: int = 1):
        if min_lineage_entries < 1:
            raise CheckConfigError("min_lineage_entries must be >= 1")
        self.min_lineage_entries = int(min_lineage_entries)

    def run(self, bundle: BundleView, evidence: QualityEvidence) -> CheckResult:
        """Report on the presence of lineage entries and an attestation."""
        lineage = bundle.lineage
        attestation = bundle.attestation
        att_fields = [
            k
            for k in ("measurement", "signature", "client_id", "verified", "quote")
            if k in attestation or k in as_mapping(attestation.get("quote"))
        ]
        has_lineage = len(lineage) >= self.min_lineage_entries
        declared_root = not has_lineage and bundle.declares_lineage_field
        details: Dict[str, Any] = {
            "n_lineage_entries": len(lineage),
            "attestation_fields": sorted(att_fields),
            "signature_verified_here": False,
            "root_submission": not has_lineage,
            "lineage_field_declared": bundle.declares_lineage_field,
        }
        if not attestation:
            missing = ["attestation"] if has_lineage else ["lineage", "attestation"]
            return self._result(
                Status.FAIL,
                f"Submission lacks {', '.join(missing)}; provenance cannot be checked.",
                details=details,
                recommendation=(
                    "Record the training run in the lineage ledger and attach the "
                    "contributor's attestation before submitting."
                ),
            )
        if not att_fields:
            return self._result(
                Status.WARN,
                "An attestation object is present but carries none of the "
                "expected fields.",
                details=details,
                recommendation="Attach the full attestation quote, not a stub.",
            )
        if not has_lineage:
            if not declared_root:
                return self._result(
                    Status.WARN,
                    "No lineage field on the bundle; treating this as a root "
                    "submission on the strength of its attestation record.",
                    details=details,
                    recommendation=(
                        "Declare the parent bundles this model was derived from, "
                        "or an empty parent list if it was trained independently."
                    ),
                )
            return self._result(
                Status.PASS,
                "Root submission: no parent lineage is declared, and an "
                "attestation record is present. Cryptographic verification is "
                "performed by the ledger and attestation components, not here.",
                details=details,
            )
        return self._result(
            Status.PASS,
            f"Lineage ({len(lineage)} entries) and an attestation record are "
            "present; cryptographic verification is performed by the ledger and "
            "attestation components, not here.",
            details=details,
        )



__all__ = ["MetadataSchemaCheck", "LineageAttestationCheck"]

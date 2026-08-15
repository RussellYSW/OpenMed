"""Sections 3 and 4: derive from the certified base, then verify the lineage."""

from __future__ import annotations

from typing import Any

from trustfed.attestation import MockSoftwareAttestor
from trustfed.registry import EvaluationReport, ModelRegistry, WeightsRef

from openmed_demo.fixtures import (
    APPROVED_PIPELINE,
    attest,
    make_card,
    make_manual,
    synthetic_metrics,
    synthetic_weights,
)


def derive(
    registry: ModelRegistry,
    attestor: MockSoftwareAttestor,
    parent_id: str,
    seed: int,
) -> Any:
    """Site B fine-tunes the certified base and publishes the derived model."""
    metrics = synthetic_metrics(seed + 1)
    derived = registry.publish_derived(
        parent_id,
        name="pd-decline",
        version="1.1.0-inst-b",
        weights=WeightsRef.from_bytes(
            "file://demo/weights/pd-decline-1.1.0.npz", synthetic_weights(seed + 1)
        ),
        model_card=make_card("pd-decline", "1.1.0-inst-b", "INST_B", metrics),
        evaluation=EvaluationReport(
            dataset_id="synthetic-holdout-inst-b",
            n_samples=420,
            metrics=metrics,
            protocol="fixed split",
            evaluated_by="INST_C",
            seed=seed + 1,
        ),
        attestation=attest(attestor, "site_b", APPROVED_PIPELINE),
        published_by="INST_B",
        fine_tuning_manual=make_manual("INST_B").to_dict(),
    )
    print(f"derived bundle          : {derived.short_id()} (v{derived.version})")
    print(f"  parent                : {registry.get(derived.parents[0]).short_id()}")
    return derived


def show_lineage(registry: ModelRegistry, bundle_id: str) -> None:
    """Verify and print the lineage of a bundle."""
    verdict = registry.verify_lineage(bundle_id)
    print(verdict.summary())
    print(f"  ledger verified       : {verdict.ledger_verified}")
    print()
    print(registry.lineage_graph(bundle_id).to_mermaid().rstrip())


__all__ = ["derive", "show_lineage"]

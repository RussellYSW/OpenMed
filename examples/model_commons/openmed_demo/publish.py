"""Section 1: an attested publish, and the refusal of an unapproved pipeline."""

from __future__ import annotations

from typing import Any

from trustfed.attestation import MockSoftwareAttestor
from trustfed.registry import EvaluationReport, ModelRegistry, RegistryError, WeightsRef

from openmed_demo.fixtures import (
    APPROVED_PIPELINE,
    UNAPPROVED_PIPELINE,
    attest,
    make_card,
    make_manual,
    synthetic_metrics,
    synthetic_weights,
)


def publish_root(
    registry: ModelRegistry, attestor: MockSoftwareAttestor, seed: int
) -> Any:
    """Site A publishes the first model, with a full card and manual."""
    metrics = synthetic_metrics(seed)
    bundle = registry.publish(
        name="pd-decline",
        version="1.0.0",
        weights=WeightsRef.from_bytes(
            "file://demo/weights/pd-decline-1.0.0.npz", synthetic_weights(seed)
        ),
        model_card=make_card("pd-decline", "1.0.0", "INST_A", metrics),
        evaluation=EvaluationReport(
            dataset_id="synthetic-holdout",
            n_samples=500,
            metrics=metrics,
            subgroup_metrics={"site_a": {"auc": round(metrics["auc"] - 0.03, 3)}},
            protocol="fixed split, seed reported in the card",
            evaluated_by="INST_B",
            seed=seed,
        ),
        attestation=attest(attestor, "site_a", APPROVED_PIPELINE),
        published_by="INST_A",
        fine_tuning_manual=make_manual("INST_A").to_dict(),
        tags=("demo", "synthetic"),
    )
    print(
        f"published root bundle   : {bundle.short_id()}  "
        f"({bundle.name} v{bundle.version})"
    )
    print(f"  attestation verified  : {bundle.attestation.verified}")
    print(f"  model card complete   : {bundle.model_card.validate().ok}")
    print(f"  manual valid          : {bundle.validate_manual().ok}")
    return bundle


def show_unapproved_publish(
    registry: ModelRegistry, attestor: MockSoftwareAttestor
) -> None:
    """A site running unapproved code cannot get a verified attestation."""
    record = attest(attestor, "site_rogue", UNAPPROVED_PIPELINE)
    print(
        f"rogue site attestation  : verified={record.verified} "
        f"reason={record.verifier_reason}"
    )
    try:
        registry.publish(
            name="pd-decline",
            version="1.0.0-rogue",
            weights=WeightsRef.from_bytes("file://rogue.npz", b"rogue"),
            model_card=make_card("pd-decline", "1.0.0-rogue", "INST_X", {"auc": 0.99}),
            evaluation=EvaluationReport(
                dataset_id="unstated", n_samples=10, metrics={"auc": 0.99}
            ),
            attestation=record,
            published_by="INST_X",
        )
    except RegistryError as exc:
        print(f"  publish refused       : {type(exc).__name__}")


__all__ = ["publish_root", "show_unapproved_publish"]

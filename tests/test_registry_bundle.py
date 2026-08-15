"""Tests for model bundles, model cards and the publish gate (Component 3).

Covers the content-addressed bundle id, the Mitchell et al. model card, and the
conditions under which the registry refuses to publish. Lineage lives in
``test_registry_lineage.py``.
"""
from __future__ import annotations

import pytest

from trustfed.attestation import AttestationPolicy, MockSoftwareAttestor, measure_code
from trustfed.ledger import HmacSigner, InMemoryLedger
from trustfed.registry import (
    AttestationRequiredError,
    BundleNotFoundError,
    DuplicateBundleError,
    EvaluationReport,
    ModelBundle,
    ModelCard,
    ModelRegistry,
    PipelineAttestation,
    ValidationError,
    WeightsRef,
)

CODE_IDENTITY = "openmed-pipeline@v1"


def make_attestor() -> MockSoftwareAttestor:
    return MockSoftwareAttestor(
        b"registry-test-root",
        policy=AttestationPolicy(approved_measurements={measure_code(CODE_IDENTITY)}),
    )


def make_attestation(site: str = "site_a", identity: str = CODE_IDENTITY):
    att = make_attestor()
    quote = att.generate_quote(site, identity, "training-config")
    return PipelineAttestation.from_quote(quote, attestor=att)


def make_card(name: str = "pd-decline", version: str = "1.0", owner: str = "INST_A"):
    return ModelCard(
        model_details={
            "name": name,
            "version": version,
            "owner": owner,
            "date": "2026-01-01",
            "model_type": "logistic regression (numpy)",
            "license": "Apache-2.0",
        },
        intended_use={"primary_use": "research prototype on synthetic cohorts"},
        factors={"groups": ["site", "age_band"]},
        metrics={"reported": ["auc", "accuracy"]},
        evaluation_data={"dataset": "synthetic held-out split"},
        training_data={"dataset": "synthetic multi-site cohort"},
        quantitative_analyses={"auc": 0.81},
        ethical_considerations={"risks": "not clinically validated"},
        caveats_and_recommendations={"caveats": "synthetic data only"},
    )


def make_eval(auc: float = 0.81):
    return EvaluationReport(
        dataset_id="synthetic-holdout",
        n_samples=500,
        metrics={"auc": auc},
        subgroup_metrics={"site_a": {"auc": auc - 0.02}},
        protocol="fixed split, seed 7",
        evaluated_by="INST_B",
        seed=7,
    )


def make_weights(blob: bytes = b"weights-v1"):
    return WeightsRef.from_bytes("file://weights/v1.npz", blob)


def publish_root(registry: ModelRegistry, **overrides):
    kwargs = dict(
        name="pd-decline",
        version="1.0",
        weights=make_weights(),
        model_card=make_card(),
        evaluation=make_eval(),
        attestation=make_attestation(),
        published_by="INST_A",
    )
    kwargs.update(overrides)
    return registry.publish(**kwargs)

# ------------------------------------------------------------------ model card


def test_model_card_has_the_nine_mitchell_sections():
    from trustfed.registry import CARD_SECTIONS

    assert CARD_SECTIONS == (
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


def test_incomplete_model_card_reports_what_is_missing():
    card = ModelCard(model_details={"name": "m"})
    result = card.validate()
    assert result.ok is False
    assert "intended_use" in result.missing_sections
    assert "version" in result.missing_details
    with pytest.raises(ValidationError):
        result.raise_if_invalid()


def test_model_card_round_trips_and_renders():
    card = make_card()
    assert ModelCard.from_dict(card.to_dict()) == card
    assert "## Model details" in card.to_markdown()
    with pytest.raises(ValidationError):
        ModelCard.from_dict({"not_a_section": {}})


def test_thin_card_produces_warnings_without_failing():
    card = make_card()
    thin = ModelCard.from_dict({**card.to_dict(), "factors": {"note": "n/a"}})
    result = thin.validate()
    assert result.ok is True
    assert any("factors.groups" in w for w in result.warnings)


# --------------------------------------------------------------------- bundles


def test_bundle_id_is_content_addressed():
    kwargs = dict(
        name="m",
        version="1.0",
        weights=make_weights(),
        model_card=make_card(),
        evaluation=make_eval(),
        attestation=make_attestation(),
        published_by="INST_A",
    )
    a = ModelBundle.create(**kwargs)
    b = ModelBundle.create(**kwargs)
    assert a.bundle_id == b.bundle_id
    assert a.bundle_id.startswith("openmed:bundle:sha256:")
    assert a.id_is_intact()

    changed = ModelBundle.create(**{**kwargs, "weights": make_weights(b"other")})
    assert changed.bundle_id != a.bundle_id


def test_tampered_bundle_content_breaks_its_id():
    bundle = ModelBundle.create(
        name="m",
        version="1.0",
        weights=make_weights(),
        model_card=make_card(),
        evaluation=make_eval(),
        attestation=make_attestation(),
        published_by="INST_A",
    )
    data = bundle.to_dict()
    data["evaluation"]["metrics"]["auc"] = 0.99  # inflate the reported metric
    rebuilt = ModelBundle.from_dict(data)
    assert rebuilt.bundle_id == bundle.bundle_id  # recorded id preserved...
    assert rebuilt.id_is_intact() is False  # ...but no longer matches content


def test_weights_and_evaluation_validate_their_inputs():
    with pytest.raises(ValidationError):
        WeightsRef(uri="", sha256="0" * 64)
    with pytest.raises(ValidationError):
        WeightsRef(uri="file://x", sha256="not-a-hash")
    with pytest.raises(ValidationError):
        EvaluationReport(dataset_id="d", n_samples=0, metrics={"auc": 0.5})
    with pytest.raises(ValidationError):
        EvaluationReport(dataset_id="d", n_samples=10, metrics={})


# -------------------------------------------------------------------- publish


def test_publish_records_the_event_on_the_ledger():
    ledger = InMemoryLedger(signer=HmacSigner(b"k"))
    registry = ModelRegistry(ledger)
    bundle = publish_root(registry)

    assert bundle.bundle_id in registry
    assert len(registry) == 1
    assert len(ledger) == 1
    payload = ledger.get(0).payload
    assert payload["event"] == "model_published"
    assert payload["bundle_id"] == bundle.bundle_id
    assert registry.ledger_records(bundle.bundle_id)[0].index == 0


def test_unverified_attestation_is_refused():
    registry = ModelRegistry()
    att = make_attestor()
    tampered_quote = att.generate_quote("site_evil", "tampered-code", "cfg")
    attestation = PipelineAttestation.from_quote(tampered_quote, attestor=att)
    assert attestation.verified is False
    with pytest.raises(AttestationRequiredError):
        publish_root(registry, attestation=attestation)
    assert len(registry) == 0
    assert len(registry.ledger) == 0


def test_incomplete_card_is_refused_at_publish():
    registry = ModelRegistry()
    with pytest.raises(ValidationError):
        publish_root(registry, model_card=ModelCard(model_details={"name": "m"}))


def test_duplicate_content_is_refused():
    """Byte-identical content hashes to the same id, so it cannot be re-published."""
    registry = ModelRegistry()
    attestation = make_attestation()
    publish_root(registry, attestation=attestation)
    with pytest.raises(DuplicateBundleError):
        publish_root(registry, attestation=attestation)
    assert len(registry) == 1


def test_publishing_with_an_unknown_parent_is_refused():
    registry = ModelRegistry()
    with pytest.raises(BundleNotFoundError):
        publish_root(registry, parents=("openmed:bundle:sha256:" + "0" * 64,))

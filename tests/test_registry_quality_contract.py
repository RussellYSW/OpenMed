"""The contract between a real registry bundle and the quality analyzer.

``trustfed.quality`` deliberately does not import ``trustfed.registry``: the
analyzer duck-types whatever it is handed, and ``tests/test_quality_report.py``
pins that decoupling. The cost of the decoupling is that nothing else pins the
*vocabulary* -- the registry says ``evaluation``/``parents``/``weights.uri``
where the analyzer's own flat form says ``metrics``/``lineage``/``weights_path``
-- so a rename on either side would break the pairing silently, and the
duck-typed stand-ins in ``tests/test_quality_bundle.py`` would keep passing.

This file is that pin. It builds an actual
:class:`trustfed.registry.ModelBundle` -- including a real
:class:`trustfed.registry.ModelCard` object rather than a plain dict -- and
asserts what the analyzer resolves from it. It is the one test that imports
both packages together, on purpose.

Ownership note: the cross-module review asked for this test and left the owner
open. Dev B (registry side) owns it; the assertions are written against the
registry vocabulary so that a registry rename is what fails here.
"""

from __future__ import annotations

import pytest

from trustfed.attestation import AttestationPolicy, MockSoftwareAttestor, measure_code
from trustfed.quality import BundleView, QualityAnalyzer, Status
from trustfed.registry import (
    EvaluationReport,
    ModelBundle,
    ModelCard,
    PipelineAttestation,
    WeightsRef,
)

CODE_IDENTITY = "openmed-pipeline@v1"


def make_attestation(site: str = "site_a") -> PipelineAttestation:
    attestor = MockSoftwareAttestor(
        b"contract-test-root",
        policy=AttestationPolicy(approved_measurements={measure_code(CODE_IDENTITY)}),
    )
    quote = attestor.generate_quote(site, CODE_IDENTITY, "training-config")
    return PipelineAttestation.from_quote(quote, attestor=attestor)


def make_card() -> ModelCard:
    """A complete Mitchell et al. card, as the registry models it."""
    return ModelCard(
        model_details={
            "name": "pd-decline",
            "version": "1.0",
            "owner": "INST_A",
            "date": "2026-01-01",
            "model_type": "logistic regression (numpy)",
            "license": "Apache-2.0",
        },
        intended_use={"primary_use": "research prototype on synthetic cohorts"},
        factors={"groups": ["site", "age_band"]},
        metrics={"reported": ["auc", "accuracy"]},
        evaluation_data={"dataset": "synthetic held-out split"},
        training_data={"dataset": "synthetic multi-site cohort"},
        quantitative_analyses={"auc": 0.81, "accuracy": 0.74},
        ethical_considerations={"risks": "not clinically validated"},
        caveats_and_recommendations={"caveats": "synthetic data only"},
    )


def make_bundle(**overrides) -> ModelBundle:
    content = dict(
        name="pd-decline",
        version="1.0",
        weights=WeightsRef.from_bytes("file://weights/v1.npz", b"weights-v1"),
        model_card=make_card(),
        evaluation=EvaluationReport(
            dataset_id="synthetic-holdout",
            n_samples=500,
            metrics={"auc": 0.81, "accuracy": 0.74},
            subgroup_metrics={
                "younger": {"auc": 0.84, "n": 250},
                "older": {"auc": 0.76, "n": 250},
            },
            protocol="fixed split, seed 7",
            evaluated_by="INST_B",
            seed=7,
        ),
        attestation=make_attestation(),
        published_by="INST_A",
    )
    content.update(overrides)
    return ModelBundle.create(**content)


def test_the_view_resolves_every_field_from_the_registry_vocabulary():
    """Each assertion names a field the registry owns; renaming one fails here."""
    bundle = make_bundle()
    view = BundleView(bundle)

    assert view.bundle_id == bundle.bundle_id
    # evaluation.metrics -> metrics, evaluation.n_samples -> n_eval
    assert view.metrics["auc"] == pytest.approx(0.81)
    assert view.metrics["accuracy"] == pytest.approx(0.74)
    assert view.metrics["n_eval"] == 500
    # evaluation.subgroup_metrics -> metrics["subgroups"]
    assert set(view.metrics["subgroups"]) == {"younger", "older"}
    # weights.uri -> weights_path
    assert "v1.npz" in str(view.weights_path)
    # A real ModelCard object, not a dict, and every section must resolve.
    assert isinstance(bundle.model_card, ModelCard)
    assert view.missing_fields() == ["lineage"]  # a root bundle has no parents
    assert view.model_card == bundle.model_card.to_dict()
    assert set(view.model_card) == set(bundle.model_card.to_dict())


def test_a_real_root_bundle_analyses_cleanly():
    """A fully populated root bundle must not be penalised for being a root."""
    report = QualityAnalyzer().analyze(make_bundle())

    assert report.result("model_card_completeness").status is Status.PASS
    assert report.result("metadata_schema").status is Status.PASS
    assert report.result("metrics_present").status is Status.PASS
    # Root submissions declare no parents; that is not a lineage failure.
    lineage = report.result("lineage_attestation")
    assert lineage.status is Status.PASS
    assert "no parent lineage" in lineage.summary
    # Two subgroups are present, so the gap screen runs rather than skipping.
    gap = report.result("subgroup_gap")
    assert gap.status is not Status.SKIP
    assert gap.details["gap"] == pytest.approx(0.08)
    # The weights live behind a locator the analyzer cannot fetch, so the norm
    # screen abstains with a stated reason instead of inventing a verdict.
    assert report.result("parameter_norm").status is Status.SKIP


def test_a_derived_bundle_carries_its_parents_into_the_lineage_check():
    parent = make_bundle()
    child = make_bundle(version="1.1", parents=(parent.bundle_id,))

    view = BundleView(child)
    assert list(view.lineage) == [parent.bundle_id]
    assert view.missing_fields() == []

    report = QualityAnalyzer().analyze(child)
    lineage = report.result("lineage_attestation")
    assert lineage.status is Status.PASS
    assert lineage.details["n_lineage_entries"] == 1
    assert lineage.details["root_submission"] is False


def test_a_registry_bundle_with_no_subgroup_evaluation_is_screened_not_skipped_silently():
    """The registry allows an evaluation without subgroups; the analyzer must
    say so rather than pass by omission."""
    bundle = make_bundle(
        evaluation=EvaluationReport(
            dataset_id="synthetic-holdout",
            n_samples=500,
            metrics={"auc": 0.81, "accuracy": 0.74},
        )
    )
    report = QualityAnalyzer().analyze(bundle)
    gap = report.result("subgroup_gap")
    assert gap.status is Status.SKIP
    assert "subgroup" in gap.summary.lower()
    assert report.result("metrics_present").status is Status.PASS


def test_the_duplicated_card_section_list_still_matches_the_registrys():
    """``trustfed/quality/checks/card.py`` copies the registry's nine section
    names as a literal to avoid a runtime import. The copy is deliberate and
    documented, but nothing stops it drifting if the registry renames or adds a
    section, so the two tuples are pinned equal here -- from the side that owns
    the original.
    """
    from trustfed.quality.checks.card import MITCHELL_CARD_SECTIONS
    from trustfed.registry.model_card import CARD_SECTIONS

    assert MITCHELL_CARD_SECTIONS == CARD_SECTIONS
    assert len(CARD_SECTIONS) == 9

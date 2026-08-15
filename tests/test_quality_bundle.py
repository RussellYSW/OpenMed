"""Tests for the defensive bundle view and the optional evidence container.

:class:`BundleView` is what stands between the quality analyzer and whatever a
submitter actually hands it, so these tests cover both vocabularies it accepts:
the analyzer's own flat dict and the registry's ``ModelBundle`` shape.
"""

from pathlib import Path

import numpy as np
import pytest

from trustfed.quality import BundleFormatError, BundleView, QualityAnalyzer, QualityEvidence, Status

GOOD_CARD = {
    "model_details": "Logistic regression, 12 features, trained federated.",
    "intended_use": "Research demonstration of federated prognosis models.",
    "out_of_scope_use": "Not for clinical decision support of any kind.",
    "training_data": "Synthetic multi-site cohort from trustfed.data.",
    "evaluation_data": "Held-out synthetic population sample, n=800.",
    "metrics": "ROC AUC and accuracy on the held-out sample.",
    "limitations": "Synthetic data only; no external validation performed.",
    "ethical_considerations": "No patient data was used at any stage.",
    "contact": "maintainers@example.invalid",
}


class FakeBundle:
    """Stand-in for the registry's bundle; the analyzer only duck-types it."""

    def __init__(self, **fields):
        for name in (
            "model_card",
            "metrics",
            "attestation",
            "lineage",
            "weights_path",
            "bundle_id",
        ):
            setattr(self, name, fields.get(name))


def good_bundle(**overrides):
    fields = {
        "bundle_id": "pd-decline-v1",
        "model_card": dict(GOOD_CARD),
        "metrics": {"auc": 0.81, "accuracy": 0.74, "n_eval": 800},
        "attestation": {"measurement": "abc123", "signature": "def456", "client_id": "site_00"},
        "lineage": [{"index": 0, "payload": "round-1"}],
        "weights_path": None,
    }
    fields.update(overrides)
    return FakeBundle(**fields)

# -- bundle view (defensive access) -----------------------------------------


def test_bundle_view_tolerates_missing_fields():
    view = BundleView(FakeBundle())
    assert view.model_card == {} and view.metrics == {}
    assert view.lineage == [] and view.weights_path is None
    assert set(view.missing_fields()) == {
        "model_card",
        "metrics",
        "attestation",
        "lineage",
        "weights_path",
    }
    assert view.bundle_id == "unidentified-bundle"


def test_bundle_view_accepts_a_plain_dict():
    view = BundleView({"model_card": {"intended_use": "demo"}, "bundle_id": "d1"})
    assert view.bundle_id == "d1"
    assert view.model_card["intended_use"] == "demo"


def test_bundle_view_rejects_uninspectable_input():
    for bad in (None, 3, "a string", True):
        with pytest.raises(BundleFormatError):
            BundleView(bad)


def test_bundle_view_normalizes_a_single_lineage_record():
    assert len(BundleView({"lineage": {"index": 0}}).lineage) == 1
    assert len(BundleView({"lineage": {"entries": [1, 2, 3]}}).lineage) == 3



def test_evidence_from_predictions_builds_subgroup_metrics():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=200)
    scores = rng.uniform(size=200)
    groups = ["a"] * 100 + ["b"] * 100
    ev = QualityEvidence.from_predictions(
        y_true=y, y_score=scores, groups=groups, train_scores=scores[:100],
        holdout_scores=scores[100:],
    )
    assert set(ev.subgroup_metrics) == {"a", "b"}
    assert ev.member_scores.shape == (100,)


# -- the registry's vocabulary ----------------------------------------------


class RegistryShapedEvaluation:
    """Duck-type of trustfed.registry.bundle.EvaluationReport."""

    def __init__(self):
        self.dataset_id = "synthetic-holdout"
        self.n_samples = 800
        self.metrics = {"auc": 0.82, "accuracy": 0.75}
        self.subgroup_metrics = {
            "younger": {"auc": 0.86, "n": 400},
            "older": {"auc": 0.71, "n": 400},
        }


class RegistryShapedWeights:
    """Duck-type of trustfed.registry.bundle.WeightsRef."""

    def __init__(self):
        self.uri = "file:///srv/models/pd-decline-v1.npz"
        self.sha256 = "0" * 64


class RegistryShapedBundle:
    """Duck-type of trustfed.registry.bundle.ModelBundle.

    It names the same facts differently: ``evaluation`` not ``metrics``,
    ``parents`` not ``lineage``, ``weights.uri`` not ``weights_path``.
    """

    def __init__(self):
        self.bundle_id = "openmed:bundle:sha256:" + "a" * 64
        self.name = "pd-rapid-decline"
        self.version = "1.0.0"
        self.weights = RegistryShapedWeights()
        self.model_card = dict(GOOD_CARD)
        self.evaluation = RegistryShapedEvaluation()
        self.attestation = {"measurement": "abc123", "signature": "def456"}
        self.parents = ("openmed:bundle:sha256:" + "b" * 64,)
        self.published_by = "INST_A"


def test_bundle_view_reads_the_registry_vocabulary():
    view = BundleView(RegistryShapedBundle())
    assert view.missing_fields() == []
    assert view.metrics["auc"] == 0.82
    assert view.metrics["n_eval"] == 800
    assert set(view.metrics["subgroups"]) == {"younger", "older"}
    assert len(view.lineage) == 1
    assert view.weights_path is not None
    assert "pd-decline-v1.npz" in str(view.weights_path)


def test_reading_the_registry_vocabulary_does_not_mutate_the_bundle():
    bundle = RegistryShapedBundle()
    BundleView(bundle).metrics["auc"] = -1.0
    assert bundle.evaluation.metrics["auc"] == 0.82


def test_analyzer_scores_a_registry_shaped_bundle_without_synthetic_help():
    # The defect this guards: a fully populated registry bundle used to produce
    # metrics_present FAIL and subgroup_gap SKIP because the field names differ.
    report = QualityAnalyzer().analyze(RegistryShapedBundle())
    assert report.result("metrics_present").status is Status.PASS
    assert report.result("subgroup_gap").status is not Status.SKIP
    assert report.result("subgroup_gap").details["gap"] == pytest.approx(0.15)
    assert report.result("metadata_schema").status is Status.PASS
    assert report.result("lineage_attestation").status is Status.PASS


def test_the_analyzer_vocabulary_still_wins_when_both_are_present():
    bundle = RegistryShapedBundle()
    bundle.metrics = {"auc": 0.5, "accuracy": 0.5}
    view = BundleView(bundle)
    assert view.metrics["auc"] == 0.5
    # ...but the evaluation report still supplies what the flat dict omits.
    assert view.metrics["n_eval"] == 800


# -- root submissions --------------------------------------------------------


def test_an_empty_parents_tuple_is_a_root_bundle_not_absent_provenance():
    # The defect this guards: every independently-trained model -- the first
    # release of any model line -- has parents=(), and used to fail the gate
    # with "Submission lacks lineage".
    bundle = RegistryShapedBundle()
    bundle.parents = ()
    view = BundleView(bundle)
    assert view.lineage == []
    assert view.declares_lineage_field is True

    report = QualityAnalyzer().analyze(bundle)
    lineage = report.result("lineage_attestation")
    assert lineage.status is not Status.FAIL
    assert lineage.status is Status.PASS
    assert lineage.details["root_submission"] is True
    schema = report.result("metadata_schema")
    assert schema.status is Status.PASS
    assert "lineage" not in schema.details["missing_fields"]


def test_a_root_bundle_without_an_attestation_still_fails():
    bundle = RegistryShapedBundle()
    bundle.parents = ()
    bundle.attestation = {}
    result = QualityAnalyzer().analyze(bundle).result("lineage_attestation")
    assert result.status is Status.FAIL
    assert "lineage" in result.summary and "attestation" in result.summary


def test_a_bundle_with_no_lineage_field_at_all_only_warns():
    # Distinct from the root case: nothing was declared either way, so the
    # analyzer says so rather than passing it silently.
    result = QualityAnalyzer().analyze(
        {
            "bundle_id": "no-lineage-field",
            "model_card": dict(GOOD_CARD),
            "metrics": {"auc": 0.8, "accuracy": 0.75, "n_eval": 500},
            "attestation": {"measurement": "abc", "client_id": "site_00"},
        }
    ).result("lineage_attestation")
    assert result.status is Status.WARN
    assert result.details["lineage_field_declared"] is False


# -- weights locators --------------------------------------------------------


def test_a_uri_locator_is_not_rewritten_as_a_filesystem_path():
    # Path("s3://bucket/x") collapses to "s3:/bucket/x"; a report that quotes
    # that back at a reviewer names an artifact the bundle never declared.
    bundle = RegistryShapedBundle()
    bundle.weights = RegistryShapedWeights()
    bundle.weights.uri = "s3://bucket/model.npz"
    assert BundleView(bundle).weights_path == "s3://bucket/model.npz"

    report = QualityAnalyzer().analyze(bundle)
    norm = report.result("parameter_norm")
    assert norm.status is Status.SKIP
    assert "s3://bucket/model.npz" in norm.summary
    assert "s3:/bucket" not in report.to_json()


def test_a_plain_filesystem_path_is_still_a_path(tmp_path):
    view = BundleView({"weights_path": str(tmp_path / "w.npz")})
    assert isinstance(view.weights_path, Path)
    assert view.weights_path.name == "w.npz"


# -- the real registry bundle (integration) ----------------------------------
#
# Every other test here duck-types the registry so the analyzer stays decoupled
# from it. That decoupling is deliberate, but nothing was pinning the shared
# vocabulary, which is how the root-bundle failure above survived. This one test
# imports the real thing and asserts the contract end to end.


def _real_root_bundle(uri: str = "file://weights/v1.npz"):
    """Build an actual trustfed.registry.ModelBundle with no parents."""
    from trustfed.attestation import AttestationPolicy, MockSoftwareAttestor, measure_code
    from trustfed.registry import (
        EvaluationReport,
        ModelBundle,
        ModelCard,
        PipelineAttestation,
        WeightsRef,
    )

    identity = "openmed-pipeline@v1"
    attestor = MockSoftwareAttestor(
        b"quality-integration-root",
        policy=AttestationPolicy(approved_measurements={measure_code(identity)}),
    )
    quote = attestor.generate_quote("site_a", identity, "training-config")
    card = ModelCard(
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
        quantitative_analyses={"auc": 0.81},
        ethical_considerations={"risks": "not clinically validated"},
        caveats_and_recommendations={"caveats": "synthetic data only"},
    )
    return ModelBundle.create(
        name="pd-decline",
        version="1.0",
        weights=WeightsRef.from_bytes(uri, b"weights-v1"),
        model_card=card,
        evaluation=EvaluationReport(
            dataset_id="synthetic-holdout",
            n_samples=500,
            metrics={"auc": 0.81, "accuracy": 0.74},
            subgroup_metrics={"younger": {"auc": 0.83}, "older": {"auc": 0.78}},
            protocol="fixed split, seed 7",
            evaluated_by="INST_B",
            seed=7,
        ),
        attestation=PipelineAttestation.from_quote(quote, attestor=attestor),
        published_by="INST_A",
    )


@pytest.mark.parametrize("serialize", [False, True])
def test_a_real_registry_root_bundle_passes_the_documentation_gate(serialize):
    bundle = _real_root_bundle()
    subject = bundle.to_dict() if serialize else bundle

    view = BundleView(subject)
    assert view.bundle_id.startswith("openmed:bundle:sha256:")
    assert view.model_card["intended_use"]
    assert view.metrics["auc"] == pytest.approx(0.81)
    assert view.metrics["n_eval"] == 500
    assert set(view.metrics["subgroups"]) == {"younger", "older"}
    assert view.attestation["measurement"]
    assert view.lineage == [] and view.declares_lineage_field is True

    report = QualityAnalyzer().analyze(subject)
    statuses = {r.check_id: r.status for r in report.results}
    assert statuses["model_card_completeness"] is Status.PASS
    assert statuses["metadata_schema"] is Status.PASS
    assert statuses["lineage_attestation"] is Status.PASS
    assert statuses["metrics_present"] is Status.PASS
    assert statuses["subgroup_gap"] is Status.PASS
    # No weights and no confidence scores were supplied, so those two screens
    # abstain rather than inventing a verdict.
    assert statuses["parameter_norm"] is Status.SKIP
    assert statuses["membership_inference"] is Status.SKIP
    assert report.overall_status is not Status.FAIL


def test_a_real_registry_bundle_keeps_its_weights_locator_intact():
    report = QualityAnalyzer().analyze(_real_root_bundle("s3://bucket/model.npz"))
    assert "s3://bucket/model.npz" in report.result("parameter_norm").summary
    assert "s3:/bucket" not in report.to_json()

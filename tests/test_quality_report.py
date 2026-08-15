"""Tests for the analyzer pipeline, report rendering and the summarizers.

Covers the roll-up of check results into a report, both serialisations, and the
deterministic template summarizer plus the optional local-LLM one.
"""

import json

import numpy as np
import pytest

from trustfed.quality import (
    BundleFormatError,
    ModelCardCompletenessCheck,
    QualityAnalyzer,
    QualityEvidence,
    Status,
    TemplateSummarizer,
    default_checks,
)
from trustfed.quality.checks.base import Check
from trustfed.quality.report import CheckResult, QualityReport
from trustfed.quality.summarize import LocalLLMSummarizer

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

# -- analyzer ---------------------------------------------------------------


def test_analyzer_runs_every_check_and_rolls_up_the_worst_status():
    report = QualityAnalyzer().analyze(good_bundle())
    assert len(report.results) == len(default_checks())
    assert report.overall_status in (Status.WARN, Status.SKIP, Status.FAIL)
    assert {r.check_id for r in report.results} >= {
        "model_card_completeness",
        "membership_inference",
        "subgroup_gap",
        "parameter_norm",
    }


def test_analyzer_works_on_a_bundle_with_nothing_in_it():
    report = QualityAnalyzer().analyze(FakeBundle())
    assert report.overall_status is Status.FAIL
    assert report.counts["fail"] >= 3
    assert report.counts["skip"] >= 1  # missing evidence is skipped, never passed


def test_analyzer_does_not_import_the_registry_package():
    import sys

    sys.modules.pop("trustfed.registry", None)
    QualityAnalyzer().analyze(good_bundle())
    assert "trustfed.registry" not in sys.modules


def test_analyzer_converts_a_broken_check_into_an_error_result():
    class ExplodingCheck(Check):
        check_id = "boom"
        title = "Exploding check"

        def run(self, bundle, evidence):
            raise RuntimeError("kaboom")

    report = QualityAnalyzer(checks=[ExplodingCheck(), ModelCardCompletenessCheck()]).analyze(
        good_bundle()
    )
    assert report.result("boom").status is Status.ERROR
    assert "kaboom" in report.result("boom").summary
    # The other check still ran.
    assert report.result("model_card_completeness").status is Status.PASS


def test_analyzer_rejects_an_uninspectable_bundle():
    with pytest.raises(BundleFormatError):
        QualityAnalyzer().analyze(None)


def test_fail_fast_stops_at_the_first_failure():
    report = QualityAnalyzer(fail_fast=True).analyze(FakeBundle())
    assert len(report.results) == 1
    assert report.results[0].status is Status.FAIL


def test_analyzer_records_which_evidence_it_was_given():
    ev = QualityEvidence(weights=np.zeros(10), notes={"source": "unit test"})
    report = QualityAnalyzer().analyze(good_bundle(), ev)
    assert report.context["evidence_supplied"] == ["weights"]
    assert report.context["evidence_notes"] == {"source": "unit test"}

# -- report rendering -------------------------------------------------------


def test_report_json_is_serializable_and_carries_the_disclaimer():
    report = QualityAnalyzer().analyze(good_bundle())
    parsed = json.loads(report.to_json())
    assert parsed["bundle_id"] == "pd-decline-v1"
    assert "does not establish" in parsed["disclaimer"]
    assert report.to_json().endswith("\n")
    assert len(parsed["results"]) == len(default_checks())


def test_report_markdown_lists_every_check_and_the_disclaimer():
    report = QualityAnalyzer().analyze(good_bundle())
    md = report.to_markdown()
    for result in report.results:
        assert result.title in md
    assert "## Checks" in md
    assert "triage aid" in md


def test_report_write_creates_both_files(tmp_path):
    report = QualityAnalyzer().analyze(good_bundle())
    written = report.write(tmp_path)
    assert written["json"].exists() and written["markdown"].exists()
    assert json.loads(written["json"].read_text(encoding="utf-8"))["overall_status"]


def test_report_handles_numpy_values_in_details(tmp_path):
    report = QualityReport(
        bundle_id="x",
        results=[
            CheckResult(
                check_id="c",
                title="C",
                status=Status.PASS,
                summary="ok",
                details={"array": np.arange(3), "scalar": np.float64(1.5), "nan": float("nan")},
            )
        ],
    )
    parsed = json.loads(report.to_json())
    assert parsed["results"][0]["details"]["array"] == [0, 1, 2]
    assert parsed["results"][0]["details"]["nan"] is None


# -- summarizers ------------------------------------------------------------


def test_template_summarizer_is_deterministic_and_mentions_failures():
    r = QualityAnalyzer().analyze(FakeBundle())
    first = TemplateSummarizer().summarize(r)
    assert first == TemplateSummarizer().summarize(r)
    assert "FAIL" in first
    assert "Model card completeness" in first


def test_template_summarizer_reports_skipped_checks_explicitly():
    r = QualityAnalyzer().analyze(good_bundle())
    text = TemplateSummarizer().summarize(r)
    assert "skipped" in text.lower()


def test_analyzer_narrates_with_the_default_summarizer_and_no_llm():
    report = QualityAnalyzer().analyze(good_bundle())
    assert report.narrative and len(report.narrative) > 50
    assert report.context["summarizer"] == "template"


def test_local_llm_summarizer_falls_back_when_the_endpoint_is_unreachable():
    summarizer = LocalLLMSummarizer("http://127.0.0.1:9/v1/chat/completions", timeout=0.2)
    report = QualityAnalyzer(summarizer=summarizer).analyze(good_bundle())
    assert report.narrative  # template fallback still produced a narrative
    assert summarizer.last_error


def test_local_llm_summarizer_refuses_a_non_loopback_endpoint():
    summarizer = LocalLLMSummarizer("http://example.invalid/v1/chat/completions")
    text = summarizer.summarize(QualityAnalyzer().analyze(good_bundle()))
    assert "allow_remote" in (summarizer.last_error or "")
    assert text  # fell back to the template rather than failing


def test_local_llm_summarizer_uses_a_stubbed_endpoint_response(monkeypatch):
    summarizer = LocalLLMSummarizer("http://localhost:11434/v1/chat/completions")
    monkeypatch.setattr(summarizer, "_request", lambda payload: "narrative from model")
    assert summarizer.summarize(QualityAnalyzer().analyze(good_bundle())) == (
        "narrative from model"
    )


def test_local_llm_summarizer_falls_back_on_an_empty_response(monkeypatch):
    summarizer = LocalLLMSummarizer("http://localhost:11434/v1/chat/completions")
    monkeypatch.setattr(summarizer, "_request", lambda payload: "   ")
    text = summarizer.summarize(QualityAnalyzer().analyze(good_bundle()))
    assert summarizer.last_error == "empty response from endpoint"
    assert "Automated analysis" in text



# -- end to end with a real trained model -----------------------------------


def test_quality_report_on_a_model_trained_by_the_federated_loop(tmp_path):
    """The analyzer should work on a bundle produced by this repo's own loop."""
    from trustfed.data import make_cohort
    from trustfed.data.cohort import CohortSpec
    from trustfed.federated import Client, Server
    from trustfed.metrics import binary_metrics, subgroup_metrics

    cohort = make_cohort(CohortSpec(n_sites=5, samples_per_site=150, seed=21))
    server = Server(cohort.n_features)
    clients = [Client(s.site_id, s.X, s.y, seed=i) for i, s in enumerate(cohort.sites)]
    server.fit(clients, cohort.X_test, cohort.y_test, rounds=20)

    test_scores = server.model.predict_proba(cohort.X_test)
    train_X = np.concatenate([s.X for s in cohort.sites])
    train_scores = server.model.predict_proba(train_X)
    overall = binary_metrics(cohort.y_test, test_scores)

    weights_path = tmp_path / "weights.npy"
    np.save(weights_path, server.model.get_params())

    bundle = FakeBundle(
        bundle_id="federated-demo",
        model_card=dict(GOOD_CARD),
        metrics={
            "auc": overall["auc"],
            "accuracy": overall["accuracy"],
            "n_eval": int(overall["n"]),
        },
        attestation={"measurement": "m", "signature": "s", "client_id": "site_00"},
        lineage=[{"round": r.round, "auc": r.auc} for r in server.history],
        weights_path=weights_path,
    )
    evidence = QualityEvidence(
        member_scores=train_scores,
        nonmember_scores=test_scores,
        subgroup_metrics=subgroup_metrics(cohort.y_test, test_scores, cohort.test_groups),
    )
    report = QualityAnalyzer().analyze(bundle, evidence)

    assert report.counts["error"] == 0
    assert report.counts["skip"] == 0  # every check had what it needed
    assert report.result("metrics_present").status is Status.PASS
    assert report.result("parameter_norm").status is Status.PASS
    # A model trained on 5 sites of synthetic data should not memorize much.
    assert report.result("membership_inference").details["membership_advantage"] < 0.2
    written = report.write(tmp_path)
    assert written["markdown"].read_text(encoding="utf-8").startswith("# Quality report")

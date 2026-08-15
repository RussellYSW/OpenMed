"""Tests for the individual quality checks.

One test class of behaviour per check, including the failure paths: a check that
cannot fail is not a gate.
"""

import numpy as np
import pytest

from trustfed.quality import (
    BundleView,
    CheckConfigError,
    LineageAttestationCheck,
    MembershipInferenceCheck,
    MetadataSchemaCheck,
    MetricsPresentCheck,
    ModelCardCompletenessCheck,
    ParameterNormCheck,
    QualityEvidence,
    Status,
    SubgroupGapCheck,
    threshold_attack_advantage,
)

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

# -- individual checks ------------------------------------------------------


def test_model_card_completeness_fails_on_an_empty_card():
    result = ModelCardCompletenessCheck().run(BundleView(FakeBundle()), QualityEvidence())
    assert result.status is Status.FAIL
    assert "model card" in result.summary.lower()


def test_model_card_completeness_rejects_placeholder_sections():
    card = dict(GOOD_CARD)
    card["limitations"] = "TODO"
    card["ethical_considerations"] = ""
    result = ModelCardCompletenessCheck().run(
        BundleView(good_bundle(model_card=card)), QualityEvidence()
    )
    assert result.status is Status.WARN
    assert set(result.details["missing_sections"]) == {
        "limitations",
        "ethical_considerations",
    }


def test_model_card_completeness_passes_a_full_card():
    result = ModelCardCompletenessCheck().run(BundleView(good_bundle()), QualityEvidence())
    assert result.status is Status.PASS
    assert result.details["completeness"] == 1.0


def test_metadata_schema_flags_missing_fields_and_bad_metric_types():
    result = MetadataSchemaCheck().run(
        BundleView(FakeBundle(model_card=dict(GOOD_CARD))), QualityEvidence()
    )
    assert result.status is Status.FAIL
    assert "weights_path" in result.details["missing_fields"]


def test_metrics_check_fails_out_of_range_values():
    result = MetricsPresentCheck().run(
        BundleView(good_bundle(metrics={"auc": 1.7, "accuracy": 0.6})), QualityEvidence()
    )
    assert result.status is Status.FAIL
    assert any("auc" in p for p in result.details["out_of_range"])


def test_metrics_check_warns_on_a_suspiciously_perfect_score():
    result = MetricsPresentCheck().run(
        BundleView(good_bundle(metrics={"auc": 1.0, "accuracy": 1.0, "n_eval": 900})),
        QualityEvidence(),
    )
    assert result.status is Status.WARN
    assert "contamination" in result.summary.lower()
    assert "disjoint" in (result.recommendation or "").lower()


def test_metrics_check_warns_on_a_tiny_evaluation_set():
    result = MetricsPresentCheck().run(
        BundleView(good_bundle(metrics={"auc": 0.8, "accuracy": 0.7, "n_eval": 12})),
        QualityEvidence(),
    )
    assert result.status is Status.WARN


def test_lineage_attestation_fails_when_provenance_is_absent():
    result = LineageAttestationCheck().run(
        BundleView(good_bundle(lineage=[], attestation=None)), QualityEvidence()
    )
    assert result.status is Status.FAIL


def test_lineage_attestation_passes_a_declared_root_submission():
    # An independently-trained model has no parents. That is a root release,
    # not withheld provenance, and the attestation is what vouches for it.
    result = LineageAttestationCheck().run(
        BundleView(good_bundle(lineage=[])), QualityEvidence()
    )
    assert result.status is Status.PASS
    assert result.details["root_submission"] is True
    assert result.details["n_lineage_entries"] == 0


def test_lineage_attestation_does_not_claim_to_verify_signatures():
    result = LineageAttestationCheck().run(BundleView(good_bundle()), QualityEvidence())
    assert result.status is Status.PASS
    assert result.details["signature_verified_here"] is False


# -- subgroup gap -----------------------------------------------------------


def test_subgroup_gap_detects_a_material_disparity():
    ev = QualityEvidence(
        subgroup_metrics={"younger": {"auc": 0.86, "n": 400}, "older": {"auc": 0.71, "n": 400}}
    )
    result = SubgroupGapCheck().run(BundleView(good_bundle()), ev)
    assert result.status is Status.FAIL
    assert result.details["worst_group"] == "older"
    assert result.details["gap"] == pytest.approx(0.15)


def test_subgroup_gap_passes_when_groups_agree():
    ev = QualityEvidence(
        subgroup_metrics={"a": {"auc": 0.80, "n": 300}, "b": {"auc": 0.79, "n": 300}}
    )
    assert SubgroupGapCheck().run(BundleView(good_bundle()), ev).status is Status.PASS


def test_subgroup_gap_excludes_groups_too_small_to_measure():
    ev = QualityEvidence(
        subgroup_metrics={
            "a": {"auc": 0.80, "n": 300},
            "b": {"auc": 0.79, "n": 300},
            "tiny": {"auc": 0.20, "n": 4},
        }
    )
    result = SubgroupGapCheck().run(BundleView(good_bundle()), ev)
    assert result.status is Status.PASS
    assert result.details["excluded_small_groups"] == ["tiny"]


def test_subgroup_gap_reads_metrics_from_the_bundle_when_no_evidence_is_given():
    bundle = good_bundle(
        metrics={
            "auc": 0.8,
            "accuracy": 0.7,
            "subgroups": {"a": {"auc": 0.9, "n": 200}, "b": {"auc": 0.6, "n": 200}},
        }
    )
    result = SubgroupGapCheck().run(BundleView(bundle), QualityEvidence())
    assert result.status is Status.FAIL


def test_subgroup_gap_skips_rather_than_passing_when_data_is_absent():
    result = SubgroupGapCheck().run(BundleView(good_bundle()), QualityEvidence())
    assert result.status is Status.SKIP


def test_subgroup_check_rejects_impossible_thresholds():
    with pytest.raises(CheckConfigError):
        SubgroupGapCheck(warn_gap=0.5, fail_gap=0.1)


# -- membership inference ---------------------------------------------------


def test_threshold_attack_finds_no_signal_in_identical_distributions():
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=500)
    auc, advantage, _ = threshold_attack_advantage(scores, rng.uniform(size=500))
    assert abs(auc - 0.5) < 0.08
    assert advantage < 0.15


def test_threshold_attack_finds_a_separated_memorizing_model():
    members = np.full(200, 0.99)
    nonmembers = np.full(200, 0.55)
    auc, advantage, threshold = threshold_attack_advantage(members, nonmembers)
    assert auc == pytest.approx(1.0)
    assert advantage == pytest.approx(1.0)
    assert 0.55 < threshold <= 0.99


def test_membership_check_fails_a_leaking_model():
    rng = np.random.default_rng(1)
    ev = QualityEvidence(
        member_scores=rng.uniform(0.9, 1.0, size=300),
        nonmember_scores=rng.uniform(0.4, 0.6, size=300),
    )
    result = MembershipInferenceCheck().run(BundleView(good_bundle()), ev)
    assert result.status is Status.FAIL
    assert result.details["membership_advantage"] > 0.2
    assert "differential privacy" in (result.recommendation or "")


def test_membership_check_passes_a_non_leaking_model():
    rng = np.random.default_rng(2)
    ev = QualityEvidence(
        member_scores=rng.uniform(0.4, 0.6, size=400),
        nonmember_scores=rng.uniform(0.4, 0.6, size=400),
    )
    result = MembershipInferenceCheck().run(BundleView(good_bundle()), ev)
    assert result.status is Status.PASS


def test_membership_check_skips_with_too_little_evidence():
    ev = QualityEvidence(member_scores=np.ones(4), nonmember_scores=np.zeros(4))
    result = MembershipInferenceCheck().run(BundleView(good_bundle()), ev)
    assert result.status is Status.SKIP
    assert MembershipInferenceCheck().run(
        BundleView(good_bundle()), QualityEvidence()
    ).status is Status.SKIP


def test_membership_check_reads_scores_from_the_bundle_metrics():
    rng = np.random.default_rng(3)
    bundle = good_bundle(
        metrics={
            "auc": 0.8,
            "accuracy": 0.7,
            "membership_inference": {
                "member_scores": rng.uniform(0.95, 1.0, size=200).tolist(),
                "nonmember_scores": rng.uniform(0.0, 0.1, size=200).tolist(),
            },
        }
    )
    assert MembershipInferenceCheck().run(BundleView(bundle), QualityEvidence()).status is Status.FAIL


# -- parameter norm ---------------------------------------------------------


def test_parameter_norm_passes_a_normal_vector():
    rng = np.random.default_rng(4)
    ev = QualityEvidence(weights=rng.normal(0.0, 1.0, size=200))
    assert ParameterNormCheck().run(BundleView(good_bundle()), ev).status is Status.PASS


def test_parameter_norm_fails_a_scaled_or_backdoored_vector():
    rng = np.random.default_rng(5)
    weights = rng.normal(0.0, 1.0, size=200)
    weights[7] = 5_000.0
    result = ParameterNormCheck().run(BundleView(good_bundle()), QualityEvidence(weights=weights))
    assert result.status is Status.FAIL
    assert result.details["max_robust_z"] > 12


def test_parameter_norm_fails_non_finite_weights():
    weights = np.ones(50)
    weights[3] = np.inf
    result = ParameterNormCheck().run(BundleView(good_bundle()), QualityEvidence(weights=weights))
    assert result.status is Status.FAIL
    assert result.details["n_nonfinite"] == 1


def test_parameter_norm_loads_weights_from_disk(tmp_path):
    path = tmp_path / "weights.npy"
    np.save(path, np.random.default_rng(6).normal(size=64))
    result = ParameterNormCheck().run(
        BundleView(good_bundle(weights_path=path)), QualityEvidence()
    )
    assert result.status is Status.PASS
    assert result.details["n_params"] == 64


def test_parameter_norm_skips_when_the_artifact_is_missing(tmp_path):
    result = ParameterNormCheck().run(
        BundleView(good_bundle(weights_path=tmp_path / "nope.npy")), QualityEvidence()
    )
    assert result.status is Status.SKIP


def test_parameter_norm_quotes_a_remote_locator_verbatim():
    # The registry stores a locator, not a local path. Reporting it through
    # pathlib would rewrite "s3://bucket/model.npz" as "s3:/bucket/model.npz".
    result = ParameterNormCheck().run(
        BundleView(good_bundle(weights_path="s3://bucket/model.npz")), QualityEvidence()
    )
    assert result.status is Status.SKIP
    assert "s3://bucket/model.npz" in result.summary



def test_parameter_norm_skips_the_coordinate_screen_on_small_models():
    # A robust z-score over 13 coordinates is noise; reporting it would fail
    # every small model. Only finiteness and total norm apply there.
    weights = np.array([2.0, -1.5, 0.9, 0.02, 0.0, 0.01, -0.03, 1.1, 0.4, 0.0, 0.0, 0.0, 0.2])
    result = ParameterNormCheck().run(BundleView(good_bundle()), QualityEvidence(weights=weights))
    assert result.status is Status.PASS
    assert result.details["coordinate_screen_applied"] is False
    assert result.details["max_robust_z"] is None


def test_parameter_norm_still_fails_a_huge_norm_on_a_small_model():
    weights = np.full(10, 1e5)
    result = ParameterNormCheck().run(BundleView(good_bundle()), QualityEvidence(weights=weights))
    assert result.status is Status.FAIL
    assert "norm" in result.summary.lower()


# -- model-card vocabularies -------------------------------------------------


def test_the_mitchell_section_copy_matches_the_registry_verbatim():
    # MITCHELL_CARD_SECTIONS is a deliberate literal copy of the registry's
    # CARD_SECTIONS (the quality package must not import the registry). This
    # pins the copy: if the registry renames or adds a section, this fails
    # instead of the copy drifting silently.
    from trustfed.quality.checks.card import MITCHELL_CARD_SECTIONS
    from trustfed.registry.model_card import CARD_SECTIONS

    assert tuple(MITCHELL_CARD_SECTIONS) == tuple(CARD_SECTIONS)


def test_a_card_that_passes_the_registry_validator_also_passes_here():
    # The defect this guards: two incompatible required-section lists meant a
    # submitter could not satisfy the registry gate and the quality gate at once.
    from trustfed.registry.model_card import CARD_SECTIONS, ModelCard

    card = ModelCard(
        model_details={
            "name": "pd-rapid-decline",
            "version": "1.0.0",
            "owner": "INST_A",
            "date": "2026-01-01",
            "model_type": "logistic regression",
            "license": "Apache-2.0",
        },
        **{
            name: {"summary": "documented for review"}
            for name in CARD_SECTIONS
            if name != "model_details"
        },
    )
    assert card.validate().ok is True

    result = ModelCardCompletenessCheck().run(
        BundleView({"model_card": card.to_dict()}), QualityEvidence()
    )
    assert result.status is Status.PASS
    assert result.details["vocabulary"] == "mitchell"


def test_the_flatter_vocabulary_is_still_accepted():
    result = ModelCardCompletenessCheck().run(
        BundleView({"model_card": dict(GOOD_CARD)}), QualityEvidence()
    )
    assert result.status is Status.PASS
    assert result.details["vocabulary"] == "trimmed"


def test_a_mitchell_card_supplies_the_trimmed_sections_from_nested_keys():
    from trustfed.quality.checks.card import section_value

    card = {
        "intended_use": {"out_of_scope": "not for clinical decision support"},
        "caveats_and_recommendations": {"limitations": "synthetic data only"},
        "model_details": {"contact": "maintainers@example.invalid"},
    }
    assert section_value(card, "out_of_scope_use") == "not for clinical decision support"
    assert section_value(card, "limitations") == "synthetic data only"
    assert section_value(card, "contact") == "maintainers@example.invalid"
    assert section_value(card, "training_data") is None


def test_an_explicit_required_list_overrides_both_vocabularies():
    check = ModelCardCompletenessCheck(required=["model_details", "nowhere_section"])
    result = check.run(BundleView({"model_card": dict(GOOD_CARD)}), QualityEvidence())
    assert result.status is Status.FAIL
    assert result.details["missing_sections"] == ["nowhere_section"]

"""Tests for the clinician-readable fine-tuning manual (Component 4).

The manual is the transfer document a receiving site reads, so its fixed schema
and its validation warnings are the behaviour under test here. Threshold rules
live in ``test_certification_threshold.py``.
"""

from __future__ import annotations

import pytest

from trustfed.certification import (
    CertificationAuthority,
    CertificationError,
    FineTuningManual,
    ManualValidationError,
    Reviewer,
    ReviewerKeyring,
    SignatureRejectedError,
    validate_manual,
)
from trustfed.certification.keys import ReviewSignature
from trustfed.certification.policy import SubmissionFacts
from trustfed.ledger import HmacSigner, InMemoryLedger

BUNDLE = "openmed:bundle:sha256:" + "ab" * 32
OWNER = "INST_A"


def _clock():
    state = {"t": 0}

    def tick() -> str:
        state["t"] += 1
        return f"2026-03-01T00:00:{state['t']:02d}.000000Z"

    return tick


def make_keyring() -> ReviewerKeyring:
    keyring = ReviewerKeyring(seed=b"cert-test")
    keyring.register(Reviewer("rev_a1", OWNER, role="clinical"))
    keyring.register(Reviewer("rev_a2", OWNER, role="ml"))
    keyring.register(Reviewer("rev_b1", "INST_B", role="clinical"))
    keyring.register(Reviewer("rev_b2", "INST_B", role="ml"))
    keyring.register(Reviewer("rev_c1", "INST_C", role="clinical"))
    keyring.register(
        Reviewer("rev_d1", "INST_D", declared_conflicts=(BUNDLE,))
    )
    return keyring


def make_authority(**kwargs) -> CertificationAuthority:
    keyring = kwargs.pop("keyring", None)
    ledger = kwargs.pop("ledger", None)
    if keyring is None:
        keyring = make_keyring()
    if ledger is None:
        ledger = InMemoryLedger(signer=HmacSigner(b"cert"))
    return CertificationAuthority(keyring, ledger, clock=_clock(), **kwargs)


def submit(authority: CertificationAuthority, **kwargs):
    params = dict(
        owner_institution=OWNER,
        submitted_by="site_a",
        contributor_institutions=(),
        contributor_ids=(),
    )
    params.update(kwargs)
    return authority.submit(BUNDLE, **params)
# -------------------------------------------------------- fine-tuning manual


def complete_manual() -> FineTuningManual:
    return FineTuningManual(
        intended_use={
            "task": "flag rapid cognitive decline within 24 months",
            "population": "adults with early-stage Parkinson's disease",
            "care_setting": "movement-disorder clinic, research use",
            "not_intended_for": "diagnosis, treatment selection, paediatric use",
        },
        data={
            "sources": ["synthetic multi-site cohort"],
            "n_records": 3000,
            "inclusion_criteria": "baseline visit plus 24-month follow-up",
            "label_definition": "decline threshold on the composite score",
        },
        preprocessing={
            "steps": ["drop incomplete visits", "z-score per site"],
            "normalization": "z-score using training-set statistics",
            "missing_data": "listwise deletion",
        },
        hyperparameters={
            "optimizer": "full-batch gradient descent",
            "learning_rate": 0.05,
            "epochs": 25,
            "batch_size": "full batch",
        },
        failure_modes={
            "known_failure_modes": ["site shift", "scanner change"],
            "monitoring": "monthly subgroup AUC review",
        },
        clinical_caveats={
            "human_oversight": "output is advisory; a clinician decides",
            "contraindications": "do not use outside the stated population",
            "escalation": "notify the site clinical lead on drift",
        },
    )


def test_complete_manual_validates_and_renders():
    manual = complete_manual()
    result = manual.validate()
    assert result.ok is True
    assert result.warnings == ()
    text = manual.to_markdown()
    assert "## What this model is for" in text
    assert "## Clinical caveats" in text
    assert FineTuningManual.from_dict(manual.to_dict()) == manual


def test_manual_reports_missing_sections_and_fields():
    manual = FineTuningManual(intended_use={"task": "t"})
    result = manual.validate()
    assert result.ok is False
    assert "data" in result.missing_sections
    assert "intended_use.population" in result.missing_fields
    assert any("failure modes" in w for w in result.warnings)
    with pytest.raises(ManualValidationError):
        result.raise_if_invalid()


def test_manual_rejects_sections_outside_the_fixed_schema():
    with pytest.raises(ManualValidationError):
        FineTuningManual.from_dict({"marketing_claims": {"best": True}})


def test_validate_manual_helper_and_required_at_submission():
    assert validate_manual(complete_manual().to_dict()).ok is True
    auth = make_authority(require_manual=True)
    with pytest.raises(CertificationError):
        submit(auth)
    incomplete = FineTuningManual(intended_use={"task": "t"}).to_dict()
    with pytest.raises(ManualValidationError):
        submit(auth, manual=incomplete)
    case = submit(auth, manual=complete_manual().to_dict())
    assert case.manual_ok is True


def test_submission_facts_are_recorded_on_the_ledger():
    auth = make_authority()
    submit(auth, contributor_institutions=("INST_B",))
    payload = auth.decision_log(BUNDLE)[0].payload
    assert payload["facts"]["owner_institution"] == OWNER
    assert payload["facts"]["contributor_institutions"] == ["INST_B"]
    facts = SubmissionFacts(BUNDLE, OWNER)
    assert facts.to_dict()["bundle_id"] == BUNDLE


def test_signature_rejected_when_key_rotates_mid_review():
    keyring = make_keyring()
    auth = make_authority(keyring=keyring)
    submit(auth)

    class _BrokenKeyring(ReviewerKeyring):
        """Keyring whose verify always fails, modelling a rotated/lost key."""

        def verify(self, signature: ReviewSignature) -> bool:
            return False

    broken = _BrokenKeyring(seed=b"cert-test")
    broken.register(Reviewer("rev_b1", "INST_B"))
    auth_broken = make_authority(keyring=broken)
    auth_broken.submit(BUNDLE, owner_institution=OWNER, submitted_by="site_a")
    with pytest.raises(SignatureRejectedError):
        auth_broken.review(BUNDLE, "rev_b1", "approve")

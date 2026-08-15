"""Tests for the k-of-n multi-institution threshold and conflict rules (Component 4).

Covers who may sign, whose signature counts, what defeats the threshold, and the
fine-tuning manual. The dispute/appeal/revocation state machine and the
append-only decision log live in ``test_certification_disputes.py``.
"""

from __future__ import annotations

import pytest

from trustfed.certification import (
    CertificationAuthority,
    CertificationError,
    CertificationState,
    ConflictOfInterestError,
    Reviewer,
    ReviewerKeyring,
    ThresholdNotMetError,
    ThresholdPolicy,
    UnknownReviewerError,
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
    return authority.submit(BUNDLE, **params)# ----------------------------------------------------------- multi-institution


def test_two_institutions_are_required_to_certify():
    auth = make_authority()
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve", statement="looks sound")

    with pytest.raises(ThresholdNotMetError) as exc:
        auth.certify(BUNDLE)
    assert "insufficient" in str(exc.value)
    assert auth.state(BUNDLE) is CertificationState.UNDER_REVIEW

    auth.review(BUNDLE, "rev_c1", "approve")
    case = auth.certify(BUNDLE)
    assert case.state is CertificationState.CERTIFIED
    assert auth.is_certified_base(BUNDLE) is True
    assert case.approving_institutions() == ("INST_B", "INST_C")


def test_two_reviewers_from_one_institution_do_not_meet_the_threshold():
    auth = make_authority()
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_b2", "approve")
    outcome = auth.evaluate(BUNDLE)
    assert outcome.met is False
    assert outcome.reason_code == "insufficient_institutions"
    with pytest.raises(ThresholdNotMetError):
        auth.certify(BUNDLE)


def test_a_site_cannot_certify_its_own_model():
    auth = make_authority()
    submit(auth)
    with pytest.raises(ConflictOfInterestError) as exc:
        auth.review(BUNDLE, "rev_a1", "approve")
    assert exc.value.reason_code == "self_certification"
    # The refusal itself is recorded, not silently dropped.
    actions = [b.payload["action"] for b in auth.decision_log(BUNDLE)]
    assert "review_refused" in actions
    assert auth.case(BUNDLE).signatures == []


def test_contributing_institution_and_personal_contribution_are_conflicts():
    auth = make_authority()
    submit(
        auth,
        contributor_institutions=("INST_B",),
        contributor_ids=("rev_c1",),
    )
    with pytest.raises(ConflictOfInterestError) as inst_exc:
        auth.review(BUNDLE, "rev_b1", "approve")
    assert inst_exc.value.reason_code == "contributing_institution"

    with pytest.raises(ConflictOfInterestError) as person_exc:
        auth.review(BUNDLE, "rev_c1", "approve")
    assert person_exc.value.reason_code == "personal_contribution"


def test_declared_conflict_is_honoured():
    auth = make_authority()
    submit(auth)
    with pytest.raises(ConflictOfInterestError) as exc:
        auth.review(BUNDLE, "rev_d1", "approve")
    assert exc.value.reason_code == "declared_conflict"


def test_unknown_reviewer_is_rejected():
    auth = make_authority()
    submit(auth)
    with pytest.raises(UnknownReviewerError):
        auth.review(BUNDLE, "nobody", "approve")


def test_revoked_reviewer_key_stops_counting():
    keyring = make_keyring()
    auth = make_authority(keyring=keyring)
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_c1", "approve")
    assert auth.evaluate(BUNDLE).met is True

    keyring.revoke("rev_c1")
    assert auth.evaluate(BUNDLE).met is False
    with pytest.raises(ThresholdNotMetError):
        auth.certify(BUNDLE)


def test_forged_review_signature_does_not_verify():
    keyring = make_keyring()
    forged = ReviewSignature(
        bundle_id=BUNDLE,
        reviewer_id="rev_b1",
        institution="INST_B",
        decision="approve",
        statement="",
        signed_at="2026-03-01T00:00:00.000000Z",
        signature="00" * 32,
        key_id="",
    )
    assert keyring.verify(forged) is False
    # Claiming a different institution than the bound one also fails.
    genuine = keyring.sign("rev_b1", forged)
    lying = ReviewSignature(
        bundle_id=genuine.bundle_id,
        reviewer_id=genuine.reviewer_id,
        institution="INST_C",
        decision=genuine.decision,
        statement=genuine.statement,
        signed_at=genuine.signed_at,
        signature=genuine.signature,
        key_id=genuine.key_id,
    )
    assert keyring.verify(genuine) is True
    assert keyring.verify(lying) is False


def test_changing_a_reviewers_mind_does_not_double_count():
    policy = ThresholdPolicy(k=2, min_institutions=2)
    auth = make_authority(policy=policy)
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_b1", "reject", statement="withdrawn")
    auth.review(BUNDLE, "rev_c1", "approve")
    outcome = auth.evaluate(BUNDLE)
    assert outcome.met is False
    assert outcome.institutions == ("INST_C",)


def test_threshold_policy_validates_its_configuration():
    with pytest.raises(ValueError):
        ThresholdPolicy(k=0)
    with pytest.raises(ValueError):
        ThresholdPolicy(k=2, min_institutions=3)
    relaxed = ThresholdPolicy(k=2, min_institutions=1, require_distinct_institutions=False)
    assert relaxed.to_dict()["require_distinct_institutions"] is False


# ------------------------------------------- facts derived from the bundle


def _bundle_parts(published_by: str = OWNER, version: str = "1.0.0"):
    """Return the keyword content of a publishable bundle owned by ``published_by``."""
    from trustfed.attestation import (
        AttestationPolicy,
        MockSoftwareAttestor,
        measure_code,
    )
    from trustfed.registry import (
        EvaluationReport,
        ModelCard,
        PipelineAttestation,
        WeightsRef,
    )

    identity = "openmed-pipeline@v1"
    attestor = MockSoftwareAttestor(
        b"cert-facts-root",
        policy=AttestationPolicy(approved_measurements={measure_code(identity)}),
    )
    quote = attestor.generate_quote(
        published_by.lower(), identity, "training-config-v1"
    )
    card = ModelCard(
        model_details={
            "name": "pd",
            "version": version,
            "owner": published_by,
            "date": "2026-01-15",
            "model_type": "logistic regression",
            "license": "Apache-2.0",
        },
        intended_use={"primary_use": "research"},
        factors={"groups": ["site"]},
        metrics={"reported": ["auc"]},
        evaluation_data={"dataset": "synthetic"},
        training_data={"dataset": "synthetic"},
        quantitative_analyses={"auc": 0.81},
        ethical_considerations={"risks": "synthetic only"},
        caveats_and_recommendations={"caveats": "demo"},
    )
    return dict(
        name="pd",
        version=version,
        weights=WeightsRef.from_bytes("file://w.npz", b"weights" + version.encode()),
        model_card=card,
        evaluation=EvaluationReport(
            dataset_id="synthetic-holdout", n_samples=100, metrics={"auc": 0.81}
        ),
        attestation=PipelineAttestation.from_quote(quote, attestor=attestor),
        published_by=published_by,
    )


def _bundle(published_by: str = OWNER, version: str = "1.0.0"):
    """Build a real, content-addressed ModelBundle owned by ``published_by``."""
    from trustfed.registry import ModelBundle

    return ModelBundle.create(**_bundle_parts(published_by, version))


def test_deriving_facts_from_the_bundle_stops_a_declared_owner_lie():
    """The registry->certification seam: published_by decides who owns a model.

    Submitting by bundle id lets a site declare any owner it likes, which
    defeats the no-self-certification rule outright. Submitting the bundle
    itself derives the owner from content the bundle id commits to.
    """
    bundle = _bundle(published_by=OWNER)

    lying = make_authority()
    lying.submit(
        bundle.bundle_id,
        owner_institution="INST_UNRELATED",
        submitted_by="site_a",
    )
    lying.review(bundle.bundle_id, "rev_a1", "approve")  # the real owner's own reviewer
    lying.review(bundle.bundle_id, "rev_b1", "approve")
    assert lying.certify(bundle.bundle_id).state is CertificationState.CERTIFIED
    assert lying.case(bundle.bundle_id).facts.self_asserted is True

    honest = make_authority()
    case = honest.submit(bundle, submitted_by="site_a")
    assert case.facts.owner_institution == OWNER
    assert case.facts.self_asserted is False
    with pytest.raises(ConflictOfInterestError) as exc:
        honest.review(bundle.bundle_id, "rev_a1", "approve")
    assert exc.value.reason_code == "self_certification"

    honest.review(bundle.bundle_id, "rev_b1", "approve")
    honest.review(bundle.bundle_id, "rev_c1", "approve")
    assert honest.certify(bundle.bundle_id).approving_institutions() == (
        "INST_B",
        "INST_C",
    )


def test_self_asserted_submissions_are_marked_on_the_decision_log():
    bundle = _bundle()
    asserted = make_authority()
    submit(asserted)
    assert asserted.decision_log(BUNDLE)[0].payload["self_asserted"] is True

    derived = make_authority()
    derived.submit(bundle, submitted_by="site_a")
    payload = derived.decision_log(bundle.bundle_id)[0].payload
    assert payload["self_asserted"] is False
    assert payload["facts"]["owner_institution"] == OWNER


def test_derived_facts_refuse_a_contradicting_declared_owner():
    bundle = _bundle()
    auth = make_authority()
    with pytest.raises(CertificationError) as exc:
        auth.submit(bundle, owner_institution="INST_Z", submitted_by="site_a")
    assert "contradicts" in str(exc.value)


def test_submitting_by_id_without_an_owner_is_refused():
    auth = make_authority()
    with pytest.raises(CertificationError):
        auth.submit(BUNDLE, submitted_by="site_a")


def test_parent_publishers_become_contributing_institutions():
    """A derived bundle's ancestors contribute, so their reviewers are conflicted."""
    from trustfed.registry import ModelRegistry

    registry = ModelRegistry(InMemoryLedger(signer=HmacSigner(b"reg")))
    root = registry.publish(**_bundle_parts(published_by="INST_B"))
    child = registry.publish_derived(
        root.bundle_id, **_bundle_parts(published_by="INST_C", version="2.0.0")
    )

    facts = SubmissionFacts.from_bundle(child, registry=registry)
    assert facts.owner_institution == "INST_C"
    assert "INST_B" in facts.contributor_institutions

    auth = make_authority()
    auth.submit(child, submitted_by="site_c", registry=registry)
    with pytest.raises(ConflictOfInterestError) as exc:
        auth.review(child.bundle_id, "rev_b1", "approve")
    assert exc.value.reason_code == "contributing_institution"


def test_from_bundle_refuses_an_object_it_cannot_attribute():
    class _Anonymous:
        bundle_id = BUNDLE
        published_by = ""
        attestation = None
        parents = ()

    with pytest.raises(CertificationError):
        SubmissionFacts.from_bundle(_Anonymous())

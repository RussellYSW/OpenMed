"""Tests for the certification state machine and the decision log (Component 4).

Covers disputes, appeals, remediation, terminal revocation, actor
authentication on the state-changing actions, and the tamper evidence of the
append-only log. Threshold and conflict rules live in
``test_certification_threshold.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustfed.certification import (
    CertificationAuthority,
    CertificationError,
    CertificationState,
    ConflictOfInterestError,
    InvalidTransitionError,
    Reviewer,
    ReviewerKeyring,
    SignatureRejectedError,
    ThresholdNotMetError,
    UnknownCaseError,
    can_transition,
    transitions_table,
)
from trustfed.certification.actions import ActionAuthorization
from trustfed.certification.keys import ReviewSignature
from trustfed.ledger import FileLedger, HmacSigner, InMemoryLedger

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

# --------------------------------------------------------------- state machine


def test_state_machine_transitions_match_the_documented_graph():
    assert can_transition(CertificationState.SUBMITTED, CertificationState.UNDER_REVIEW)
    assert can_transition(CertificationState.UNDER_REVIEW, CertificationState.CERTIFIED)
    assert can_transition(CertificationState.CERTIFIED, CertificationState.BLOCKED)
    assert can_transition(CertificationState.BLOCKED, CertificationState.REMEDIATED)
    assert can_transition(CertificationState.REMEDIATED, CertificationState.UNDER_REVIEW)
    assert not can_transition(CertificationState.SUBMITTED, CertificationState.CERTIFIED)
    assert not can_transition(CertificationState.REVOKED, CertificationState.CERTIFIED)
    assert dict(transitions_table())["revoked"] == ()


def test_dispute_blocks_a_certified_model_and_appeal_restores_review():
    auth = make_authority()
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_c1", "approve")
    auth.certify(BUNDLE)

    auth.dispute(BUNDLE, raised_by="INST_C", reason="subgroup gap found post hoc")
    assert auth.state(BUNDLE) is CertificationState.BLOCKED
    assert auth.is_certified_base(BUNDLE) is False

    auth.appeal(BUNDLE, actor="site_a", grounds="gap was a data error")
    auth.resolve_appeal(BUNDLE, actor="authority", upheld=True, note="re-review")
    assert auth.state(BUNDLE) is CertificationState.REMEDIATED
    assert auth.case(BUNDLE).signatures == []  # prior approvals cleared

    auth.review(BUNDLE, "rev_b1", "approve")
    assert auth.state(BUNDLE) is CertificationState.UNDER_REVIEW
    auth.review(BUNDLE, "rev_c1", "approve")
    auth.certify(BUNDLE)
    assert auth.state(BUNDLE) is CertificationState.CERTIFIED


def test_refused_appeal_revokes_terminally():
    auth = make_authority()
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "block", statement="unsafe subgroup behaviour")
    assert auth.state(BUNDLE) is CertificationState.BLOCKED

    auth.appeal(BUNDLE, actor="site_a", grounds="disagree")
    auth.resolve_appeal(BUNDLE, actor="authority", upheld=False, note="objection stands")
    assert auth.state(BUNDLE) is CertificationState.REVOKED
    with pytest.raises(InvalidTransitionError):
        auth.remediate(BUNDLE, actor="site_a", note="too late")
    with pytest.raises(InvalidTransitionError):
        auth.certify(BUNDLE)


def test_blocking_review_prevents_certification():
    auth = make_authority()
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_c1", "block", statement="evaluation protocol unclear")
    outcome = auth.evaluate(BUNDLE)
    assert outcome.met is False
    assert outcome.reason_code == "blocking_review"
    assert outcome.blocking_reviewers == ("rev_c1",)


def test_appeal_only_applies_to_blocked_cases():
    auth = make_authority()
    submit(auth)
    with pytest.raises(CertificationError):
        auth.appeal(BUNDLE, actor="site_a", grounds="premature")


def test_duplicate_submission_and_unknown_case_are_typed_errors():
    auth = make_authority()
    submit(auth)
    with pytest.raises(CertificationError):
        submit(auth)
    with pytest.raises(UnknownCaseError):
        auth.case("openmed:bundle:sha256:" + "00" * 32)
    with pytest.raises(CertificationError):
        auth.review(BUNDLE, "rev_b1", "maybe")


def test_revocation_of_a_certified_model():
    auth = make_authority()
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_c1", "approve")
    auth.certify(BUNDLE)
    auth.revoke(BUNDLE, actor="authority", reason="withdrawn by owner")
    assert auth.state(BUNDLE) is CertificationState.REVOKED
    assert auth.is_certified_base(BUNDLE) is False


# ----------------------------------------------------------- append-only log


def test_every_decision_is_recorded_in_order():
    auth = make_authority()
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_c1", "approve")
    auth.certify(BUNDLE)
    actions = [b.payload["action"] for b in auth.decision_log(BUNDLE)]
    assert actions == [
        "submitted",
        "review_opened",
        "review_signed",
        "review_signed",
        "certified",
    ]
    assert auth.verify_log().ok is True


def test_certification_cannot_be_retroactively_edited(tmp_path: Path):
    """Rewriting a refusal into an approval breaks the decision log's chain."""
    path = tmp_path / "decisions.jsonl"
    ledger = FileLedger(path, signer=HmacSigner(b"cert"))
    auth = make_authority(ledger=ledger)
    submit(auth)
    with pytest.raises(ConflictOfInterestError):
        auth.review(BUNDLE, "rev_a1", "approve")
    auth.review(BUNDLE, "rev_b1", "approve")
    assert auth.verify_log().ok is True

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])  # the recorded refusal
    assert record["payload"]["action"] == "review_refused"
    record["payload"]["action"] = "review_signed"
    record["payload"]["reason_code"] = "ok"
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = auth.verify_log()
    assert result.ok is False
    assert "payload_mutated" in result.codes


def test_refused_certification_is_also_logged():
    auth = make_authority()
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    with pytest.raises(ThresholdNotMetError):
        auth.certify(BUNDLE)
    actions = [b.payload["action"] for b in auth.decision_log(BUNDLE)]
    assert "certification_refused" in actions


# ------------------------------------------------- who may move the state machine


def _blocking_signature(keyring: ReviewerKeyring, reviewer_id: str, institution: str):
    """Return a verified blocking *review* signature from ``reviewer_id``."""
    return keyring.sign(
        reviewer_id,
        ReviewSignature(
            bundle_id=BUNDLE,
            reviewer_id=reviewer_id,
            institution=institution,
            decision="block",
            statement="withdraw this model",
            signed_at="2026-03-01T00:00:10.000000Z",
            signature="",
            key_id="",
        ),
    )


def _authorization(
    keyring: ReviewerKeyring,
    reviewer_id: str,
    action: str,
    *,
    nonce: str = "nonce-1",
) -> ActionAuthorization:
    """Return a verified action authorisation from ``reviewer_id`` over BUNDLE."""
    return keyring.sign_action(
        reviewer_id,
        ActionAuthorization(
            bundle_id=BUNDLE,
            actor=reviewer_id,
            action=action,
            nonce=nonce,
            signed_at="2026-03-01T00:00:10.000000Z",
        ),
    )


def _certified(auth: CertificationAuthority) -> None:
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_c1", "approve")
    auth.certify(BUNDLE)


def test_an_unsigned_revocation_is_recorded_as_unauthenticated():
    """revoke() is terminal, so the log must not imply the actor was checked."""
    auth = make_authority()
    _certified(auth)
    auth.revoke(BUNDLE, actor="whoever", reason="withdrawn")
    entry = auth.decision_log(BUNDLE)[-1].payload
    assert entry["action"] == "revoked"
    assert entry["actor"] == "whoever"
    assert entry["actor_authenticated"] is False
    assert entry["actor_institution"] is None


def test_an_unregistered_actor_is_visible_in_the_log():
    auth = make_authority()
    _certified(auth)
    auth.dispute(BUNDLE, raised_by="rev_c1", reason="subgroup gap")
    entry = auth.decision_log(BUNDLE)[-1].payload
    assert entry["actor_authenticated"] is False
    assert entry["actor_institution"] == "INST_C"  # name resolves, but nothing signed


def test_a_signed_revocation_is_verified_and_recorded_as_authenticated():
    keyring = make_keyring()
    auth = make_authority(keyring=keyring)
    _certified(auth)
    signed = _authorization(keyring, "rev_c1", "revoke")
    auth.revoke(BUNDLE, actor="rev_c1", reason="withdrawn", authorization=signed)
    entry = auth.decision_log(BUNDLE)[-1].payload
    assert entry["actor_authenticated"] is True
    assert entry["actor_institution"] == "INST_C"
    assert entry["authorization"]["action"] == "revoke"
    assert auth.state(BUNDLE) is CertificationState.REVOKED


def test_require_signed_actions_refuses_an_unsigned_revocation():
    keyring = make_keyring()
    auth = make_authority(keyring=keyring, require_signed_actions=True)
    _certified(auth)
    with pytest.raises(SignatureRejectedError):
        auth.revoke(BUNDLE, actor="rev_c1", reason="withdrawn")
    assert auth.state(BUNDLE) is CertificationState.CERTIFIED

    auth.revoke(
        BUNDLE,
        actor="rev_c1",
        reason="withdrawn",
        authorization=_authorization(keyring, "rev_c1", "revoke"),
    )
    assert auth.state(BUNDLE) is CertificationState.REVOKED


def test_a_forged_or_mismatched_authorization_cannot_revoke():
    keyring = make_keyring()
    auth = make_authority(keyring=keyring, require_signed_actions=True)
    _certified(auth)

    forged = ActionAuthorization(
        bundle_id=BUNDLE,
        actor="rev_c1",
        action="revoke",
        nonce="n",
        signed_at="2026-03-01T00:00:10.000000Z",
        signature="00" * 32,
    )
    with pytest.raises(SignatureRejectedError):
        auth.revoke(BUNDLE, actor="rev_c1", reason="x", authorization=forged)

    # A genuine authorisation from somebody else is not this actor's.
    other = _authorization(keyring, "rev_b1", "revoke")
    with pytest.raises(SignatureRejectedError):
        auth.revoke(BUNDLE, actor="rev_c1", reason="x", authorization=other)

    # An authorisation for a recoverable action is not authority to revoke.
    lesser = _authorization(keyring, "rev_c1", "dispute")
    with pytest.raises(SignatureRejectedError) as exc:
        auth.revoke(BUNDLE, actor="rev_c1", reason="x", authorization=lesser)
    assert "dispute" in str(exc.value)

    # And one covering a different bundle does not cover this one.
    wrong_bundle = keyring.sign_action(
        "rev_c1",
        ActionAuthorization(
            bundle_id="openmed:bundle:sha256:" + "cd" * 32,
            actor="rev_c1",
            action="revoke",
            nonce="n2",
            signed_at="2026-03-01T00:00:10.000000Z",
        ),
    )
    with pytest.raises(SignatureRejectedError):
        auth.revoke(BUNDLE, actor="rev_c1", reason="x", authorization=wrong_bundle)

    assert auth.state(BUNDLE) is CertificationState.CERTIFIED


def test_a_published_review_signature_cannot_be_replayed_as_an_action():
    """The blocker: a filed blocking review is public in the decision log.

    Lifting it out of the log and presenting it as authority to revoke -- or to
    refuse an appeal, which is equally terminal -- must fail, and must fail
    even when the authority was built to require signed actions.
    """
    keyring = make_keyring()
    auth = make_authority(keyring=keyring, require_signed_actions=True)
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")

    # rev_c1 files an honest blocking review through the supported path.
    filed = auth.submit_signature(_blocking_signature(keyring, "rev_c1", "INST_C"))
    assert auth.state(BUNDLE) is CertificationState.BLOCKED

    # An attacker reads the signature straight out of the published log.
    published = [
        b.payload["signature"]
        for b in auth.decision_log(BUNDLE)
        if b.payload.get("action") == "review_signed"
    ]
    lifted = ReviewSignature.from_dict(published[-1])
    assert lifted == filed
    assert keyring.verify(lifted) is True  # it really does still verify

    with pytest.raises(SignatureRejectedError) as exc:
        auth.revoke(
            BUNDLE, actor="rev_c1", reason="attacker text", authorization=lifted
        )
    assert "ActionAuthorization" in str(exc.value)
    assert auth.state(BUNDLE) is CertificationState.BLOCKED

    auth.appeal(BUNDLE, actor="site_a", grounds="disagree")
    with pytest.raises(SignatureRejectedError):
        auth.resolve_appeal(
            BUNDLE, actor="rev_c1", upheld=False, authorization=lifted
        )
    assert auth.state(BUNDLE) is CertificationState.BLOCKED


def test_an_action_authorization_is_single_use():
    """An authorisation is published in the log too, so it must not be reusable."""
    keyring = make_keyring()
    auth = make_authority(keyring=keyring, require_signed_actions=True)
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")

    once = _authorization(keyring, "rev_c1", "dispute")
    auth.dispute(BUNDLE, raised_by="rev_c1", reason="subgroup gap", authorization=once)
    assert auth.state(BUNDLE) is CertificationState.BLOCKED

    auth.remediate(BUNDLE, actor="site_a", note="fixed")
    auth.review(BUNDLE, "rev_b1", "approve")
    with pytest.raises(SignatureRejectedError) as exc:
        auth.dispute(
            BUNDLE, raised_by="rev_c1", reason="again", authorization=once
        )
    assert "already" in str(exc.value)
    assert auth.state(BUNDLE) is CertificationState.UNDER_REVIEW

    # A freshly signed one works, so the rule refuses reuse and not the actor.
    again = _authorization(keyring, "rev_c1", "dispute", nonce="nonce-2")
    auth.dispute(BUNDLE, raised_by="rev_c1", reason="again", authorization=again)
    assert auth.state(BUNDLE) is CertificationState.BLOCKED


def test_a_replay_is_still_refused_by_an_authority_rebuilt_over_the_same_ledger():
    """The spent-token set is in memory; the decision log outlives the process."""
    keyring = make_keyring()
    ledger = InMemoryLedger(signer=HmacSigner(b"cert"))
    auth = make_authority(keyring=keyring, ledger=ledger)
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    spent = _authorization(keyring, "rev_c1", "dispute")
    auth.dispute(BUNDLE, raised_by="rev_c1", reason="gap", authorization=spent)

    restarted = CertificationAuthority(keyring, ledger, clock=_clock())
    submit(restarted)
    with pytest.raises(SignatureRejectedError) as exc:
        restarted.dispute(BUNDLE, raised_by="rev_c1", reason="gap", authorization=spent)
    assert "decision log" in str(exc.value)


def test_a_signed_blocking_review_blocks_a_certified_model():
    """Post-certification safety signals belong on the authenticated path."""
    keyring = make_keyring()
    auth = make_authority(keyring=keyring)
    _certified(auth)
    assert auth.state(BUNDLE) is CertificationState.CERTIFIED

    auth.submit_signature(_blocking_signature(keyring, "rev_b2", "INST_B"))
    assert auth.state(BUNDLE) is CertificationState.BLOCKED

    entries = [b.payload for b in auth.decision_log(BUNDLE)]
    signed = [e for e in entries if e["action"] == "review_signed"][-1]
    assert signed["actor"] == "rev_b2"
    assert signed["signature"]["institution"] == "INST_B"
    blocked = entries[-1]
    assert blocked["action"] == "blocked_by_review"
    assert blocked["from_state"] == "certified"
    assert blocked["to_state"] == "blocked"
    # And the case is recoverable from there, exactly as a dispute would be.
    assert auth.evaluate(BUNDLE).reason_code == "blocking_review"


def test_require_signed_actions_also_covers_a_refused_appeal():
    keyring = make_keyring()
    auth = make_authority(keyring=keyring, require_signed_actions=True)
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "block", statement="unsafe")
    auth.appeal(BUNDLE, actor="site_a", grounds="disagree")
    with pytest.raises(SignatureRejectedError):
        auth.resolve_appeal(BUNDLE, actor="rev_b1", upheld=False)
    assert auth.state(BUNDLE) is CertificationState.BLOCKED

    # An authorisation for a different action does not unlock the refusal.
    with pytest.raises(SignatureRejectedError):
        auth.resolve_appeal(
            BUNDLE,
            actor="rev_b1",
            upheld=False,
            authorization=_authorization(keyring, "rev_b1", "remediate"),
        )
    assert auth.state(BUNDLE) is CertificationState.BLOCKED

    # Upholding an appeal is recoverable, so it is not gated the same way.
    auth.resolve_appeal(BUNDLE, actor="rev_b1", upheld=True, note="re-review")
    assert auth.state(BUNDLE) is CertificationState.REMEDIATED

    # A matching authorisation does refuse an appeal, and that is terminal.
    auth.review(BUNDLE, "rev_b1", "block", statement="still unsafe")
    auth.appeal(BUNDLE, actor="site_a", grounds="disagree again")
    auth.resolve_appeal(
        BUNDLE,
        actor="rev_b1",
        upheld=False,
        authorization=_authorization(keyring, "rev_b1", "appeal_refused"),
    )
    assert auth.state(BUNDLE) is CertificationState.REVOKED

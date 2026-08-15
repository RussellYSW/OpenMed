"""Tests for externally produced review signatures and verify-only keyrings.

The threshold is only a control if the signatures counting toward it were made
by someone other than the process counting them. These tests exercise
``submit_signature`` -- the path where the authority holds no private key at
all. Threshold arithmetic lives in ``test_certification_threshold.py``.
"""

from __future__ import annotations

import pytest

from trustfed.certification import (
    CertificationAuthority,
    CertificationState,
    ConflictOfInterestError,
    KeyringReadOnlyError,
    Reviewer,
    ReviewerKeyring,
    SignatureRejectedError,
)
from trustfed.certification.keys import ReviewSignature
from trustfed.ledger import Ed25519Signer, HmacSigner, InMemoryLedger

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
    return authority.submit(BUNDLE, **params)# ----------------------------------------------- externally produced signatures


def test_detached_signature_is_accepted_without_the_authority_holding_a_key():
    """The deployment path: reviewers sign elsewhere, the authority only verifies."""
    pytest.importorskip("cryptography")
    reviewer_keys = {
        "rev_b1": Ed25519Signer.from_seed(b"rev_b1"),
        "rev_c1": Ed25519Signer.from_seed(b"rev_c1"),
    }
    signing_ring = ReviewerKeyring(seed=b"detached")
    signing_ring.register(Reviewer("rev_b1", "INST_B"), reviewer_keys["rev_b1"])
    signing_ring.register(Reviewer("rev_c1", "INST_C"), reviewer_keys["rev_c1"])

    verifying = signing_ring.verify_only()
    assert verifying.can_sign is False
    with pytest.raises(KeyringReadOnlyError):
        verifying.sign("rev_b1", ReviewSignature(BUNDLE, "rev_b1", "INST_B", "approve", "", "t", "", ""))

    auth = make_authority(keyring=verifying)
    auth.submit(BUNDLE, owner_institution=OWNER, submitted_by="site_a")
    for reviewer_id, institution in (("rev_b1", "INST_B"), ("rev_c1", "INST_C")):
        remote = signing_ring.sign(
            reviewer_id,
            ReviewSignature(
                bundle_id=BUNDLE,
                reviewer_id=reviewer_id,
                institution=institution,
                decision="approve",
                statement="reviewed offline",
                signed_at="2026-03-01T00:00:00.000000Z",
                signature="",
                key_id="",
            ),
        )
        auth.submit_signature(remote)
    assert auth.certify(BUNDLE).state is CertificationState.CERTIFIED


def test_detached_signature_that_does_not_verify_is_rejected():
    auth = make_authority()
    submit(auth)
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
    with pytest.raises(SignatureRejectedError):
        auth.submit_signature(forged)
    assert auth.case(BUNDLE).signatures == []
    assert auth.state(BUNDLE) is CertificationState.SUBMITTED


def test_detached_signature_still_obeys_the_conflict_rules():
    keyring = make_keyring()
    auth = make_authority(keyring=keyring)
    submit(auth)
    owner_side = keyring.sign(
        "rev_a1",
        ReviewSignature(
            bundle_id=BUNDLE,
            reviewer_id="rev_a1",
            institution=OWNER,
            decision="approve",
            statement="",
            signed_at="2026-03-01T00:00:00.000000Z",
            signature="",
            key_id="",
        ),
    )
    with pytest.raises(ConflictOfInterestError):
        auth.submit_signature(owner_side)


def test_verify_only_keyring_keeps_bindings_and_revocations():
    keyring = make_keyring()
    keyring.revoke("rev_c1")
    verifying = keyring.verify_only()
    assert verifying.is_revoked("rev_c1") is True
    assert verifying.institution_of("rev_b1") == "INST_B"
    assert len(verifying) == len(keyring)
    with pytest.raises(KeyringReadOnlyError):
        verifying.register(Reviewer("rev_new", "INST_E"))


def test_the_case_snapshot_and_the_threshold_agree_after_a_key_revocation():
    """A revoked reviewer stops counting -- in both places or in neither.

    ``evaluate()`` drops signatures that no longer verify. If the case snapshot
    published to callers did not, the two would disagree: the JSON would list an
    approving institution the threshold refuses to count.
    """
    keyring = make_keyring()
    auth = make_authority(keyring=keyring)
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_c1", "approve")

    case = auth.case(BUNDLE)
    assert case.to_dict()["approving_institutions"] == ["INST_B", "INST_C"]
    assert list(auth.evaluate(BUNDLE).institutions) == ["INST_B", "INST_C"]

    keyring.revoke("rev_c1")
    outcome = auth.evaluate(BUNDLE)
    assert outcome.met is False
    assert list(outcome.institutions) == ["INST_B"]
    assert case.to_dict()["approving_institutions"] == list(outcome.institutions)
    # The signature itself is still on file; it simply no longer counts.
    assert len(case.to_dict()["signatures"]) == 2


def test_a_signed_block_moves_a_certified_case_without_a_dispute_call():
    """The strongest safety signal must be available on the signed path."""
    keyring = make_keyring()
    auth = make_authority(keyring=keyring)
    submit(auth)
    auth.review(BUNDLE, "rev_b1", "approve")
    auth.review(BUNDLE, "rev_c1", "approve")
    auth.certify(BUNDLE)

    blocking = keyring.sign(
        "rev_b2",
        ReviewSignature(
            bundle_id=BUNDLE,
            reviewer_id="rev_b2",
            institution="INST_B",
            decision="block",
            statement="subgroup harm found in deployment",
            signed_at="2026-03-01T00:01:00.000000Z",
            signature="",
            key_id="",
        ),
    )
    auth.submit_signature(blocking)
    assert auth.state(BUNDLE) is CertificationState.BLOCKED

    payloads = [b.payload for b in auth.decision_log(BUNDLE)]
    signed = [p for p in payloads if p["action"] == "review_signed"][-1]
    assert signed["signature"]["reviewer_id"] == "rev_b2"
    assert signed["signature"]["decision"] == "block"
    assert payloads[-1]["from_state"] == "certified"
    assert payloads[-1]["to_state"] == "blocked"

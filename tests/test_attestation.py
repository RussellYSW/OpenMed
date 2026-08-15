"""Tests for the attestation admission gate, including replay and policy paths."""

from __future__ import annotations

import pytest

from trustfed.attestation import (
    AttestationError,
    AttestationPolicy,
    Attestor,
    MockSoftwareAttestor,
    NonceError,
    NonceStore,
    PolicyError,
    Quote,
    measure_code,
)
from trustfed.federated.client import APPROVED_CODE_IDENTITY

ROOT_KEY = b"unit-test-root-key"
APPROVED = measure_code(APPROVED_CODE_IDENTITY)


def make_attestor():
    return MockSoftwareAttestor(ROOT_KEY, {APPROVED})


class _FakeClock:
    """Manually advanced clock so freshness tests are deterministic."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------- original public API (v1)


def test_valid_quote_verifies():
    att = make_attestor()
    q = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "config-A")
    assert att.verify(q) is True


def test_tampered_code_is_rejected():
    att = make_attestor()
    q = att.generate_quote("site_evil", "trustfed-client@TAMPERED", "config-A")
    # Signature is valid but the measurement is not on the approved allow-list.
    assert att.verify(q) is False


def test_forged_signature_is_rejected():
    att = make_attestor()
    forged = Quote(
        client_id="site_00",
        measurement=APPROVED,
        config_hash=MockSoftwareAttestor.config_hash("config-A"),
        signature="deadbeef" * 8,
    )
    assert att.verify(forged) is False


def test_wrong_root_key_cannot_forge():
    signer = MockSoftwareAttestor(b"attacker-key", {APPROVED})
    verifier = make_attestor()
    q = signer.generate_quote("site_00", APPROVED_CODE_IDENTITY, "config-A")
    assert verifier.verify(q) is False


# --------------------------------------------------------- structured verdicts


def test_check_returns_machine_readable_reason_codes():
    att = make_attestor()
    good = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert att.check(good).reason_code == "ok"
    assert att.check(good).to_dict()["ok"] is True

    bad_code = att.generate_quote("site_evil", "tampered", "cfg")
    assert att.check(bad_code).reason_code == "unapproved_measurement"

    forged = Quote("site_00", APPROVED, MockSoftwareAttestor.config_hash("cfg"), "00")
    assert att.check(forged).reason_code == "bad_signature"


def test_verify_or_raise_raises_attestation_error():
    att = make_attestor()
    bad = att.generate_quote("site_evil", "tampered", "cfg")
    with pytest.raises(AttestationError):
        att.verify_or_raise(bad)
    good = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert att.verify_or_raise(good) is good


# ---------------------------------------------------------------------- policy


def test_policy_allow_list_and_revocation():
    policy = AttestationPolicy.from_code_identities([APPROVED_CODE_IDENTITY])
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy)
    q = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert att.verify(q) is True

    att.set_policy(policy.without_measurements(APPROVED))
    assert att.verify(q) is False  # revoking the measurement locks the client out

    att.set_policy(att.policy.with_measurements(APPROVED))
    assert att.verify(q) is True


def test_empty_policy_fails_closed():
    att = MockSoftwareAttestor(ROOT_KEY, policy=AttestationPolicy())
    q = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert att.check(q).reason_code == "unapproved_measurement"


def test_config_allow_list_is_enforced():
    policy = AttestationPolicy(
        approved_measurements={APPROVED},
        approved_config_hashes={MockSoftwareAttestor.config_hash("cfg-A")},
    )
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy)
    assert att.verify(att.generate_quote("s", APPROVED_CODE_IDENTITY, "cfg-A")) is True
    result = att.check(att.generate_quote("s", APPROVED_CODE_IDENTITY, "cfg-B"))
    assert result.reason_code == "unapproved_config"


def test_policy_is_immutable_and_validates():
    policy = AttestationPolicy(approved_measurements={APPROVED})
    with pytest.raises(AttributeError):
        policy.require_nonce = True  # type: ignore[misc]
    with pytest.raises(PolicyError):
        AttestationPolicy(max_age_seconds=0)
    with pytest.raises(PolicyError):
        AttestationPolicy(clock_skew_seconds=-1)
    with pytest.raises(PolicyError):
        MockSoftwareAttestor(ROOT_KEY, {APPROVED}, policy=policy)
    with pytest.raises(PolicyError):
        MockSoftwareAttestor()


# ------------------------------------------------------------------- freshness


def test_stale_quote_is_rejected():
    clock = _FakeClock()
    policy = AttestationPolicy(approved_measurements={APPROVED}, max_age_seconds=60)
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, clock=clock)
    q = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert att.verify(q) is True
    clock.advance(61)
    result = att.check(q)
    assert result.ok is False
    assert result.reason_code == "quote_expired"


def test_quote_from_the_future_is_rejected():
    clock = _FakeClock()
    policy = AttestationPolicy(
        approved_measurements={APPROVED}, max_age_seconds=60, clock_skew_seconds=5
    )
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, clock=clock)
    q = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert att.check(q, now=clock.now - 30).reason_code == "quote_from_future"
    assert att.check(q, now=clock.now - 2).ok is True  # within skew tolerance


# ----------------------------------------------------------- nonce / anti-replay


def test_replayed_quote_is_refused():
    clock = _FakeClock()
    store = NonceStore(clock=clock)
    policy = AttestationPolicy(approved_measurements={APPROVED}, require_nonce=True)
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, nonce_store=store, clock=clock)

    nonce = att.issue_nonce("site_00")
    quote = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg", nonce=nonce)
    assert att.check(quote, expected_nonce=nonce).ok is True

    # An eavesdropper re-presents the very same quote.
    replay = att.check(quote, expected_nonce=nonce)
    assert replay.ok is False
    assert replay.reason_code == "nonce_rejected"


def test_requiring_a_nonce_without_a_challenge_store_is_refused():
    """The misconfiguration must fail closed, not fail open.

    A verifier with no NonceStore cannot issue challenges, so it could only
    check that *some* nonce is present -- one the prover invented. That accepts
    a replay forever while looking like anti-replay is switched on, so the
    combination is refused at construction, and again at check time if the
    store is taken away afterwards.
    """
    policy = AttestationPolicy(approved_measurements={APPROVED}, require_nonce=True)
    with pytest.raises(PolicyError) as exc:
        MockSoftwareAttestor(ROOT_KEY, policy=policy)
    assert "NonceStore" in str(exc.value)

    # Nor by switching a nonce-free attestor onto a nonce-requiring policy.
    plain = MockSoftwareAttestor(ROOT_KEY, {APPROVED})
    with pytest.raises(PolicyError):
        plain.set_policy(policy)

    # And check() refuses rather than accepting a prover-chosen nonce if the
    # store is removed behind the policy's back.
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, nonce_store=NonceStore())
    quote = att.generate_quote(
        "site_00", APPROVED_CODE_IDENTITY, "cfg", nonce="attacker-chosen-nonce"
    )
    att._nonces = None
    result = att.check(quote)
    assert result.ok is False
    assert result.reason_code == "nonce_rejected"
    assert "no challenge store" in result.detail


def test_quote_without_nonce_rejected_when_policy_requires_one():
    policy = AttestationPolicy(approved_measurements={APPROVED}, require_nonce=True)
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, nonce_store=NonceStore())
    q = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert att.check(q).reason_code == "missing_nonce"


def test_quote_answering_a_different_challenge_is_rejected():
    store = NonceStore()
    policy = AttestationPolicy(approved_measurements={APPROVED}, require_nonce=True)
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, nonce_store=store)
    mine = att.issue_nonce("site_00")
    other = att.issue_nonce("site_00")
    q = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg", nonce=other)
    assert att.check(q, expected_nonce=mine).reason_code == "nonce_mismatch"


def test_nonce_is_bound_to_the_requesting_client():
    store = NonceStore()
    nonce = store.issue("site_00")
    with pytest.raises(NonceError):
        store.consume("site_01", nonce)
    store.consume("site_00", nonce)
    with pytest.raises(NonceError):
        store.consume("site_00", nonce)


def test_nonce_expires():
    clock = _FakeClock()
    store = NonceStore(ttl_seconds=10, clock=clock)
    nonce = store.issue("site_00")
    assert store.is_outstanding(nonce) is True
    clock.advance(11)
    with pytest.raises(NonceError):
        store.consume("site_00", nonce)


def test_nonce_is_covered_by_the_signature():
    """Swapping the nonce on a captured quote invalidates the signature."""
    store = NonceStore()
    policy = AttestationPolicy(approved_measurements={APPROVED}, require_nonce=True)
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, nonce_store=store)
    fresh = att.issue_nonce("site_00")
    captured = att.generate_quote(
        "site_00", APPROVED_CODE_IDENTITY, "cfg", nonce="old-challenge"
    )
    tampered = Quote(
        client_id=captured.client_id,
        measurement=captured.measurement,
        config_hash=captured.config_hash,
        signature=captured.signature,
        nonce=fresh,
        issued_at=captured.issued_at,
        attestor_id=captured.attestor_id,
    )
    assert att.check(tampered, expected_nonce=fresh).reason_code == "bad_signature"


def test_attestor_without_nonce_store_raises():
    att = make_attestor()
    with pytest.raises(PolicyError):
        att.issue_nonce("site_00")


# ------------------------------------------------------------ pluggable backend


def test_ed25519_backend_lets_verifiers_check_without_forging():
    pytest.importorskip("cryptography")
    from trustfed.ledger.crypto import Ed25519Signer, Ed25519Verifier

    signer = Ed25519Signer.from_seed(b"attestation-root")
    prover = MockSoftwareAttestor(signer=signer, policy=AttestationPolicy({APPROVED}))
    relying_party = MockSoftwareAttestor(
        signer=signer,
        verifier=Ed25519Verifier.from_public_hex(signer.key_id),
        policy=AttestationPolicy({APPROVED}),
    )
    q = prover.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert relying_party.verify(q) is True

    impostor = MockSoftwareAttestor(
        signer=Ed25519Signer.from_seed(b"not-the-root"),
        policy=AttestationPolicy({APPROVED}),
    )
    forged = impostor.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert relying_party.check(forged).reason_code == "bad_signature"


def test_custom_backend_only_needs_two_methods():
    """A minimal Attestor subclass still works with the shared helpers."""

    class AlwaysDeny(Attestor):
        def generate_quote(self, client_id, code_identity, config, *, nonce=""):
            return Quote(client_id, measure_code(code_identity), "", "", nonce)

        def verify(self, quote, **kwargs):
            return False

    att = AlwaysDeny()
    q = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    assert att.check(q).ok is False
    with pytest.raises(AttestationError):
        att.verify_or_raise(q)
    with pytest.raises(PolicyError):
        att.issue_nonce("site_00")


def test_quote_serialises_round_trip():
    att = make_attestor()
    q = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")
    data = q.to_dict()
    assert data["client_id"] == "site_00"
    assert Quote(**data) == q


# ------------------------------------------- verdict ordering and nonce spending


def test_future_dated_quote_is_rejected_even_without_a_max_age():
    """Staleness and clock-lying are different questions; one guard must not hide
    the other. A policy that sets no maximum age has not agreed to accept a
    quote timestamped next year."""
    clock = _FakeClock()
    policy = AttestationPolicy(
        approved_measurements={APPROVED}, max_age_seconds=None, clock_skew_seconds=5
    )
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, clock=clock)
    quote = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg")

    far_future = att.check(quote, now=clock.now - 86_400)
    assert far_future.ok is False
    assert far_future.reason_code == "quote_from_future"
    assert att.check(quote, now=clock.now - 2).ok is True  # inside the skew window
    # An old quote is still fine: this policy asked for no freshness bound.
    assert att.check(quote, now=clock.now + 86_400).ok is True


def test_an_untimestamped_quote_is_exempt_from_the_future_check():
    """issued_at == 0 means the two-field protocol, not 1970."""
    policy = AttestationPolicy(approved_measurements={APPROVED})
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy)
    legacy = Quote(
        client_id="site_00",
        measurement=APPROVED,
        config_hash=MockSoftwareAttestor.config_hash("cfg"),
        signature="",
    )
    signed = Quote(
        client_id=legacy.client_id,
        measurement=legacy.measurement,
        config_hash=legacy.config_hash,
        signature=att._signer.sign(legacy.signing_material()),
    )
    assert att.check(signed, now=1_000.0).ok is True


def test_a_rejected_quote_does_not_spend_the_challenge():
    """A revoked code version must not burn a fresh challenge on every attempt,
    and an attacker must not be able to spend a victim's outstanding one."""
    store = NonceStore()
    policy = AttestationPolicy(approved_measurements={APPROVED}, require_nonce=True)
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, nonce_store=store)

    nonce = att.issue_nonce("site_00")
    bad = att.generate_quote("site_00", "patched-locally@v9", "cfg", nonce=nonce)
    result = att.check(bad, expected_nonce=nonce)
    assert result.reason_code == "unapproved_measurement"
    assert store.is_outstanding(nonce) is True

    # The same challenge still works for a quote that is otherwise acceptable.
    good = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg", nonce=nonce)
    assert att.check(good, expected_nonce=nonce).ok is True
    assert store.is_outstanding(nonce) is False


def test_an_unapproved_config_also_leaves_the_challenge_outstanding():
    store = NonceStore()
    policy = AttestationPolicy(
        approved_measurements={APPROVED},
        approved_config_hashes={MockSoftwareAttestor.config_hash("cfg-approved")},
        require_nonce=True,
    )
    att = MockSoftwareAttestor(ROOT_KEY, policy=policy, nonce_store=store)
    nonce = att.issue_nonce("site_00")
    quote = att.generate_quote("site_00", APPROVED_CODE_IDENTITY, "cfg-other", nonce=nonce)
    assert att.check(quote, expected_nonce=nonce).reason_code == "unapproved_config"
    assert store.is_outstanding(nonce) is True

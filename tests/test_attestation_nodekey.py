"""NodeKeyAttestor: quotes signed at the client with a registered key."""

from __future__ import annotations

import pytest

from trustfed.attestation import (
    AttestationError,
    AttestationPolicy,
    NodeKeyAttestor,
    NonceStore,
    Quote,
    software_measurement,
)
from trustfed.ledger.crypto import Ed25519Signer, HAVE_CRYPTOGRAPHY, HmacSigner
from trustfed.registry import PipelineAttestation


def _signer(label: str):
    return Ed25519Signer.from_seed(label.encode()) if HAVE_CRYPTOGRAPHY else HmacSigner(label.encode(), key_label=label)


def _attestor(measurement: str, **kwargs):
    policy = AttestationPolicy(approved_measurements={measurement}, require_nonce=True, max_age_seconds=60)
    return NodeKeyAttestor(policy=policy, nonce_store=NonceStore(), **kwargs)


def test_measurement_is_reproducible_and_changes_with_code(tmp_path):
    a = software_measurement()
    b = software_measurement()
    assert a.measurement == b.measurement
    assert a.manifest["packages"][0]["package"] == "trustfed"
    assert a.manifest["packages"][0]["files"] > 20
    script = tmp_path / "train.py"
    script.write_text("print('v1')")
    c = software_measurement(script=script)
    script.write_text("print('v2')")
    d = software_measurement(script=script)
    assert c.measurement != a.measurement
    assert c.measurement != d.measurement


def test_client_signed_quote_is_accepted_and_bound_to_the_nonce():
    measurement = software_measurement().measurement
    attestor = _attestor(measurement)
    site = _signer("site-a")
    attestor.register_client("site-a", site.verifier())
    nonce = attestor.issue_nonce("site-a")
    quote = NodeKeyAttestor.sign_quote(site, client_id="site-a", measurement=measurement, config_hash="cfg", nonce=nonce)
    record = PipelineAttestation.from_quote(quote, attestor=attestor, expected_nonce=nonce)
    assert record.verified and record.verifier_reason == "ok"
    # Replay is refused: the challenge was spent.
    assert attestor.check(quote, expected_nonce=nonce).reason_code == "nonce_rejected"


def test_wrong_key_unknown_client_and_unapproved_measurement_are_refused():
    measurement = software_measurement().measurement
    attestor = _attestor(measurement)
    site = _signer("site-a")
    attestor.register_client("site-a", site.verifier())
    nonce = attestor.issue_nonce("site-a")
    forged = NodeKeyAttestor.sign_quote(_signer("mallory"), client_id="site-a", measurement=measurement, config_hash="cfg", nonce=nonce)
    assert attestor.check(forged, expected_nonce=nonce).reason_code == "bad_signature"
    stranger = NodeKeyAttestor.sign_quote(site, client_id="site-z", measurement=measurement, config_hash="cfg", nonce=nonce)
    assert attestor.check(stranger, expected_nonce=nonce).reason_code == "unknown_client"
    tampered = NodeKeyAttestor.sign_quote(site, client_id="site-a", measurement="ff" * 32, config_hash="cfg", nonce=nonce)
    assert attestor.check(tampered, expected_nonce=nonce).reason_code == "unapproved_measurement"
    # A refused quote does not spend the challenge; the honest one still works.
    honest = NodeKeyAttestor.sign_quote(site, client_id="site-a", measurement=measurement, config_hash="cfg", nonce=nonce)
    assert attestor.check(honest, expected_nonce=nonce).ok


def test_mock_quotes_and_verifier_generation_are_refused():
    measurement = software_measurement().measurement
    attestor = _attestor(measurement)
    site = _signer("site-a")
    attestor.register_client("site-a", site.verifier())
    nonce = attestor.issue_nonce("site-a")
    mock_style = Quote(client_id="site-a", measurement=measurement, config_hash="cfg", signature="00", nonce=nonce, issued_at=1.0)
    assert attestor.check(mock_style, expected_nonce=nonce).reason_code == "wrong_attestor"
    with pytest.raises(AttestationError):
        attestor.generate_quote("site-a", "anything", "cfg")


def test_key_rotation_and_stale_quotes():
    measurement = software_measurement().measurement
    clock = {"t": 1000.0}
    attestor = _attestor(measurement, clock=lambda: clock["t"])
    old, new = _signer("old"), _signer("new")
    attestor.register_client("site-a", old.verifier())
    nonce = attestor.issue_nonce("site-a")
    quote = NodeKeyAttestor.sign_quote(old, client_id="site-a", measurement=measurement, config_hash="c", nonce=nonce, issued_at=1000.0)
    attestor.register_client("site-a", new.verifier())  # rotated before verification
    assert attestor.check(quote, expected_nonce=nonce).reason_code == "bad_signature"
    fresh = NodeKeyAttestor.sign_quote(new, client_id="site-a", measurement=measurement, config_hash="c", nonce=nonce, issued_at=1000.0)
    clock["t"] = 1100.0
    assert attestor.check(fresh, expected_nonce=nonce).reason_code == "quote_expired"

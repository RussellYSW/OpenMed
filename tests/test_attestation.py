from trustfed.attestation.attestor import MockSoftwareAttestor, Quote, measure_code
from trustfed.federated.client import APPROVED_CODE_IDENTITY

ROOT_KEY = b"unit-test-root-key"
APPROVED = measure_code(APPROVED_CODE_IDENTITY)


def make_attestor():
    return MockSoftwareAttestor(ROOT_KEY, {APPROVED})


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
